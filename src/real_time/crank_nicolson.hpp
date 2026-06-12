/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__REAL_TIME__CRANK_NICOLSON
#define INQ__REAL_TIME__CRANK_NICOLSON

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <observables/density.hpp>
#include <operations/preconditioner.hpp>
#include <solvers/steepest_descent.hpp>
#include <solvers/bicgstab.hpp>
#include <systems/electrons.hpp>
#include <systems/ions.hpp>
#include <utils/profiling.hpp>

namespace inq {
namespace real_time {

template <class OpType>
class crank_nicolson_op {

	OpType & op_;
	complex factor_;
	bool parallel_transport_;
	std::vector<matrix::distributed<complex>> overlap_;
	states::index orbital_index_;
	
public:
	
	crank_nicolson_op(OpType & op, complex factor, bool parallel_transport):
		op_(op),
		factor_(factor),
		parallel_transport_(parallel_transport)
	{
	}


	template <typename Electrons>
	void update(Electrons const & electrons) {

		CALI_CXX_MARK_SCOPE("crank_nicolson_op::update");

		if(not parallel_transport_) return;
		
		auto iphi = 0;
		for(auto & phi : electrons.kpin()) {
			orbital_index_[phi.key()] = iphi;
			auto opphi = op_(phi);
			overlap_.emplace_back(operations::overlap(opphi, phi));
		}
	}
	
	template <class PhiType>
	PhiType operator()(PhiType const & phi, double sign = 1.0) const {

		CALI_CXX_MARK_SCOPE("crank_nicolson_op");
		
		auto opphi = op_(phi);

		if(parallel_transport_) {
			assert(overlap_.size() > 0u);
			auto index = orbital_index_.at(phi.key());
			operations::rotate(overlap_[index], phi, opphi, -1.0, 1.0);
		}

		gpu::run(opphi.local_set_size(), opphi.basis().local_size(),
							 [_opphi = begin(opphi.matrix()), _phi = begin(phi.matrix()), _factor = sign*factor_] GPU_LAMBDA (auto ist, auto ip){
								 _opphi[ip][ist] = _phi[ip][ist] + _factor*_opphi[ip][ist];
							 });

		return opphi;
	}
};

template <class IonSubPropagator, class ForcesType, class HamiltonianType, class SelfConsistencyType, class EnergyType>
void crank_nicolson(double const time, double const dt, systems::ions & ions, systems::electrons & electrons, IonSubPropagator const & ion_propagator,
										ForcesType const & forces, HamiltonianType & ham, SelfConsistencyType & sc, EnergyType & energy, bool const parallel_transport,
										double const dens_tol, double const exxe_tol){

	assert(not sc.has_induced_vector_potential());

	CALI_CXX_MARK_FUNCTION;

	operations::preconditioner prec;
	//	operations::no_preconditioner prec;
	
	crank_nicolson_op<decltype(ham)> op{ham, complex{0.0, 0.5*dt}, parallel_transport};
	op.update(electrons);

	//calculate the right hand side with H(t)
	std::vector<states::orbital_set<basis::fourier_space, complex>> rhs;
	rhs.reserve(electrons.kpin_size());	
	for(auto & phi : electrons.kpin()) rhs.emplace_back(operations::transform::to_fourier(op(phi, /*sign = */ - 1.0)));
	
	//propagate ionic positions to t + dt
	ion_propagator.propagate_positions(dt, ions, forces);
	if(not ion_propagator.static_ions()) {
		sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot());
		ham.update_projectors(electrons.states_basis(), electrons.atomic_pot(), ions);
		energy.ion(ionic::interaction_energy(ions.cell(), ions, electrons.atomic_pot()));
	}

	using mix_arr_type = std::remove_reference_t<decltype(electrons.spin_density())>;
	auto mixer = mixers::broyden<mix_arr_type>(4, 0.3, electrons.spin_density().matrix().flatted().size());

	auto old_exxe = 0.0;
	auto const update_hf_max = 5;
	auto update_hf_count = 0;
	auto exxe_diff = 1.0;
	
	double density_diff;

	//now calculate the wave functions in t + dt by solving a self-consistent linear equation
	for(int istep = 0; istep < 300; istep++) {
		CALI_CXX_MARK_SCOPE("crank_nicolson:iteration");

		sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
		op.update(electrons);
		
		if(update_hf_count <= 0) {
			auto exxe = sc.update_exchange(ham, energy, electrons);
			exxe_diff = fabs(exxe - old_exxe);
			old_exxe = exxe;
			update_hf_count = update_hf_max;
		} else {
			update_hf_count--;
		}
		
		auto res = 0.0;
		auto iphi = 0;
		for(auto & phi : electrons.kpin()){
			auto fphi = operations::transform::to_fourier(std::move(phi));
			auto ires = solvers::bicgstab(fphi.set_comm(), op, prec, rhs[iphi], fphi);
			phi = operations::transform::to_real(std::move(fphi));
			operations::orthogonalize(phi);
			res += ires*electrons.kpin_weights()[iphi];
			iphi++;
		}
		
		if(electrons.kpin_states_comm().size() > 1) electrons.kpin_states_comm().all_reduce_in_place_n(&res, 1, std::plus<>{});
		
		auto new_density = observables::density::calculate(electrons);
		auto qq = observables::density::normalize(new_density, electrons.states().num_electrons());
			
		density_diff = operations::integral_sum_absdiff(electrons.spin_density(), new_density)/electrons.states().num_electrons();
		
		if(electrons.full_comm().root()) {
			std::cout << istep << '\t' << qq << '\t' << density_diff << '\t' << res;
			if(sc.has_exact_exchange()) std::cout << '\t' << exxe_diff;
			std::cout << std::endl;
		}
		
		if(density_diff < dens_tol) {
			update_hf_count = 0;
			if(exxe_diff < exxe_tol) break;
		}

		for(auto & phi : electrons.kpin()) operations::orthogonalize(phi);
		
		mixer(electrons.spin_density(), new_density);
		observables::density::normalize(electrons.spin_density(), electrons.states().num_electrons());
	}

	if(density_diff > dens_tol) throw std::runtime_error("INQ: Crank-Nicolson SCF did not converge");

	electrons.spin_density() = observables::density::calculate(electrons);
	sc.update_hamiltonian(ham, energy, electrons.spin_density(), time + dt);
	op.update(electrons);
	sc.update_exchange(ham, energy, electrons);
	 
}

}
}
#endif

#ifdef INQ_REAL_TIME_CRANK_NICOLSON_UNIT_TEST
#undef INQ_REAL_TIME_CRANK_NICOLSON_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
