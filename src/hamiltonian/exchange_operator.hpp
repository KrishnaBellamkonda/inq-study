/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__EXCHANGE_OPERATOR
#define INQ__HAMILTONIAN__EXCHANGE_OPERATOR

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/real_space.hpp>
#include <hamiltonian/singularity_correction.hpp>
#include <input/parallelization.hpp>
#include <matrix/diagonal.hpp>
#include <matrix/cholesky.hpp>
#include <operations/overlap.hpp>
#include <operations/overlap_diagonal.hpp>
#include <operations/rotate.hpp>
#include <parallel/arbitrary_partition.hpp>
#include <parallel/array_iterator.hpp>
#include <solvers/poisson.hpp>
#include <states/index.hpp>
#include <states/orbital_set.hpp>

#include <optional>

namespace inq {
namespace hamiltonian {
  class exchange_operator {
		
		gpu::array<double, 1> occupations_;
		gpu::array<vector3<double, covariant>, 1> kpoints_;
		gpu::array<int, 1> kpoint_indices_;
		mutable std::optional<basis::field_set<basis::real_space, complex, parallel::arbitrary_partition>> orbitals_;
		double exchange_coefficient_;
		std::optional<singularity_correction> sing_;
		
  public:

		exchange_operator(systems::cell const & cell, ionic::brillouin const & bzone, double const exchange_coefficient):
			exchange_coefficient_(exchange_coefficient) {
			if(enabled()) sing_.emplace(cell, bzone);
		}
			
		//////////////////////////////////////////////////////////////////////////////////

		template <class ElectronsType>
		double update(ElectronsType const & el, std::vector<states::orbital_set<basis::real_space, complex>> & ace_orbitals, states::index & orbital_index_) {
			if(not enabled()) return 0.0;

			CALI_CXX_MARK_SCOPE("exchange_operator::update");

			assert(input::parallelization::dimension_domains() == 0);
			assert(input::parallelization::dimension_states()  == 1);
			assert(input::parallelization::dimension_kpoints() == 2);

			auto shape = el.full_comm().shape();
			auto merged_comm = parallel::cartesian_communicator<2>(el.full_comm(), {shape[0], shape[1]*shape[2]});

			if(el.full_comm().coordinates()[0] != merged_comm.coordinates()[0]) {
				std::logic_error("INQ: failed flatten of communicator.");
			}

			auto part = parallel::arbitrary_partition(el.max_local_set_size()*el.kpin_size(), merged_comm.axis(1));

			occupations_ = el.occupations().flatted();
			kpoints_.reextent(part.local_size());
			kpoint_indices_.reextent(part.local_size());

			if(not orbitals_.has_value()) orbitals_.emplace(el.states_basis(), part, merged_comm);

			{
				auto ist = 0;
				for(auto & phi : el.kpin()){
					
					gpu::run(phi.local_set_size(), [ist, kpoints = begin(kpoints_), indices = begin(kpoint_indices_), kp = phi.kpoint(), ind = el.kpoint_index(phi)] GPU_LAMBDA (auto ii) {
						kpoints[ist + ii] = kp;
						indices[ist + ii] = ind;
					});

					orbitals_->matrix()({0, phi.basis().local_size()}, {ist, ist + phi.local_set_size()}) = phi.matrix();
					
					ist += phi.local_set_size();
				}
			}

			ace_orbitals.clear();
			
			auto energy = 0.0;
			{
				auto iphi = 0;
				for(auto & phi : el.kpin()){
					
					orbital_index_[phi.key()] = iphi;
					
					auto exxphi = direct(phi, -1.0);
					auto exx_matrix = operations::overlap(exxphi, phi);
					
					energy += -0.5*real(operations::sum_product(el.occupations()[iphi], matrix::diagonal(exx_matrix)));
					
					matrix::cholesky(exx_matrix);
					operations::rotate_trs(exx_matrix, exxphi);
					
					ace_orbitals.emplace_back(std::move(exxphi));
					
					iphi++;
				}
			}
				
			orbitals_.reset();
			
			el.kpin_states_comm().all_reduce_in_place_n(&energy, 1);

			return energy;
		}

		//////////////////////////////////////////////////////////////////////////////////
#ifndef ENABLE_GPU
	private:
#endif
		
		auto direct(const states::orbital_set<basis::real_space, complex> & phi, double scale = 1.0) const {
			states::orbital_set<basis::real_space, complex> exxphi(phi.skeleton());
			exxphi.fill(0.0);
			direct(phi, exxphi, scale);
			return exxphi;
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		template <class HFType, class HFOccType, class KptType, class IdxType, class PhiType, class ExxphiType>
		void block_exchange(double factor, HFType const & hf, HFOccType const & hfocc, KptType const & kpt, IdxType const & idx, PhiType const & phi, ExxphiType & exxphi) const {

			auto nst = phi.local_set_size();
			auto nhf = (~hf).size();
			basis::field_set<basis::real_space, complex> rhoij(phi.basis(), nst);
			
			for(int jj = 0; jj < nhf; jj++){

				if(fabs(hfocc[jj]) < 1e-10) continue;
				
				{ CALI_CXX_MARK_SCOPE("exchange_operator::generate_density");
					gpu::run(nst, phi.basis().local_size(),
									 [rho = begin(rhoij.matrix()), hfo = begin(hf), ph = begin(phi.matrix()), jj] GPU_LAMBDA (auto ist, auto ipoint){ 
										 rho[ipoint][ist] = conj(hfo[ipoint][jj])*ph[ipoint][ist];
									 });
				}

				solvers::poisson::in_place(rhoij, -phi.kpoint() + kpt[jj], (*sing_)(idx[jj]));
				
				{ CALI_CXX_MARK_SCOPE("exchange_operator::mulitplication");
					gpu::run(nst, exxphi.basis().local_size(),
									 [pot = begin(rhoij.matrix()), hfo = begin(hf), exph = begin(exxphi.matrix()), scal = factor*hfocc[jj], jj]
									 GPU_LAMBDA (auto ist, auto ipoint){
										 exph[ipoint][ist] += scal*hfo[ipoint][jj]*pot[ipoint][ist];
									 });
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		void direct(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi, double scale = 1.0) const {
			if(not enabled()) return;
			
			CALI_CXX_MARK_SCOPE("exchange_operator::direct");

			assert(orbitals_.has_value());
			
			double factor = -0.5*scale*exchange_coefficient_;

			if(not orbitals_->set_part().parallel()){
				block_exchange(factor, orbitals_->matrix(), occupations_, kpoints_, kpoint_indices_, phi, exxphi);
			} else {
				auto occ_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), occupations_);
				auto kpt_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), kpoints_);
				auto idx_it = parallel::array_iterator(orbitals_->set_part(), orbitals_->set_comm(), kpoint_indices_);
				for(auto ipart = 0; ipart < orbitals_->set_comm().size(); ipart++) {
					block_exchange(factor, orbitals_->matrix(), *occ_it, *kpt_it, *idx_it, phi, exxphi);
					orbitals_->shift_states();
					++occ_it;
					++kpt_it;
					++idx_it;
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////////////
		
		bool enabled() const {
			return fabs(exchange_coefficient_) > 1.0e-14;
		}

  };

}
}
#endif

#ifdef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST
#undef INQ_HAMILTONIAN_EXCHANGE_OPERATOR_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;
	
}
#endif
