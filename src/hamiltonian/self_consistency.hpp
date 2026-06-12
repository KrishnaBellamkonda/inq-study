/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__KS_POTENTIAL
#define INQ__HAMILTONIAN__KS_POTENTIAL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <basis/field.hpp>
#include <solvers/poisson.hpp>
#include <observables/density.hpp>
#include <operations/add.hpp>
#include <operations/integral.hpp>
#include <hamiltonian/xc_term.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/zeeman_coupling.hpp>
#include <hamiltonian/paw.hpp>
#include <options/theory.hpp>
#include <perturbations/none.hpp>
#include <solvers/velocity_verlet.hpp>
#include <utils/profiling.hpp>
#include <utils/vec_field.hpp>

namespace inq {
namespace hamiltonian {

template <typename Perturbation = perturbations::none>
class self_consistency {
	
	options::theory theory_;
	hamiltonian::xc_term xc_;
	std::optional<hamiltonian::xc_term> gs_xc_;
	basis::field<basis::real_space, double> vion_;
	basis::field<basis::real_space, double> core_density_;
	basis::real_space potential_basis_;
	basis::real_space density_basis_;
	Perturbation pert_;
	vector3<double, covariant> induced_vector_potential_vel_;
	vector3<double, covariant> induced_vector_potential_;
	mutable exchange_operator exact_exchange_; //this is mutable for now, we need to sort this out
	double gs_exc_;
	double gs_nvxc_;
	std::optional<basis::field_set<basis::real_space, double>> gs_vxc_;

public:

	template <typename Electrons>
	self_consistency(options::theory theory, Electrons const & electrons, Perturbation const & pert = {}, options::theory const & gs_theory = {}):
		theory_(theory),
		xc_(theory, electrons.states().num_density_components()),
		vion_(electrons.density_basis()),
		core_density_(electrons.density_basis()),
		potential_basis_(electrons.states_basis()),
		density_basis_(electrons.density_basis()),
		pert_(pert),
		induced_vector_potential_vel_({0.0, 0.0, 0.0}),
		induced_vector_potential_({0.0, 0.0, 0.0}),
		exact_exchange_(electrons.states_basis().cell(), electrons.brillouin_zone(), exx_coefficient()),
		gs_exc_(0.0),
		gs_nvxc_(0.0)
	{
		if(gs_theory.has_value()) {
			gs_xc_.emplace(gs_theory, electrons.states().num_density_components());

			assert(fabs(exx_coefficient()) < 1e-15);
			
			if(gs_xc_->exchange().true_functional()) {
				assert(fabs(gs_xc_->exchange().exx_coefficient()) < 1e-15);
			} else {
				assert(fabs(gs_theory.exchange_coefficient()) < 1e-15);
			}
		}
		
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	self_consistency(self_consistency && old, parallel::communicator new_comm):
		theory_(std::move(old.theory_)),
		xc_(std::move(old.xc_)),
		vion_(std::move(old.vion_), new_comm),
		core_density_(std::move(old.core_density_), new_comm),
		potential_basis_(std::move(old.potential_basis_), new_comm),
		density_basis_(std::move(old.density_basis_), new_comm),
		pert_(std::move(old.pert_)),
		induced_vector_potential_vel_(old.induced_vector_potential_vel_),
		induced_vector_potential_(old.induced_vector_potential_)
	{
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	template <class CommType, class ions_type>
	void update_ionic_fields(CommType & comm, ions_type const & ions, hamiltonian::atomic_potential const & atomic_pot) {

		CALI_CXX_MARK_FUNCTION;

		auto ionic_long_range  = solvers::poisson::solve(atomic_pot.ionic_density(comm, density_basis_, ions));
		auto ionic_short_range = atomic_pot.local_potential(comm, density_basis_, ions);
		vion_ = operations::add(ionic_long_range, ionic_short_range);

		core_density_ = atomic_pot.nlcc_density(comm, density_basis_, ions);
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <class CommType, class ions_type>
	void update_ionic_fields(CommType & comm, ions_type const & ions, hamiltonian::atomic_potential const & atomic_pot, std::optional<hamiltonian::paw> & pawinfo) {

		CALI_CXX_MARK_FUNCTION;

		if (pawinfo) {
			pawinfo->calculate_vloc_ncoret(ions);

			assert(pawinfo->vloc_sum().basis() == vion_.basis());
			vion_ = pawinfo->vloc_sum();

			assert(pawinfo->ncoret_sum().basis() == core_density_.basis());
			core_density_ = pawinfo->ncoret_sum();
		} else {
			update_ionic_fields(comm, ions, atomic_pot);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Electrons>
	void time_zero(Electrons const & electrons) {
		if(not gs_xc_.has_value()) return;
		
		std::optional<basis::field_set<basis::real_space, double>> kinetic_energy_density, vmgga;
		assert(not xc_.any_requires_kinetic_energy_density());
		assert(not gs_xc_->any_requires_kinetic_energy_density());
		
		gs_vxc_.emplace(density_basis_, electrons.spin_density().set_size());
		gs_xc_->operator()(electrons.spin_density(), core_density_, kinetic_energy_density, *gs_vxc_, vmgga, gs_exc_, gs_nvxc_);

		basis::field_set<basis::real_space, double> vxc(density_basis_, electrons.spin_density().set_size());
		double exc, nvxc;
		xc_(electrons.spin_density(), core_density_, kinetic_energy_density, vxc, vmgga, exc, nvxc);

		gpu::run(gs_vxc_->local_set_size(), gs_vxc_->basis().local_size(),
						 [_gs_vxc = gs_vxc_->matrix().begin(), _vxc = vxc.matrix().cbegin()] GPU_LAMBDA (auto is, auto ip){
							 _gs_vxc[ip][is] -= _vxc[ip][is];
						 });

		gs_exc_  -= exc;
		gs_nvxc_ -= nvxc;
		
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto requires_kinetic_energy_density() const {
		return xc_.any_requires_kinetic_energy_density();
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename HamiltonianType, typename EnergyType, typename FieldType>
	void update_hamiltonian(HamiltonianType & hamiltonian, EnergyType & energy, FieldType const & spin_density, std::optional<FieldType> const & kinetic_energy_density, std::optional<FieldType> const & spin_comp_charge, double time = 0.0) const {

		CALI_CXX_MARK_FUNCTION;

		assert(spin_density.basis() == density_basis_);
		assert(core_density_.basis() == spin_density.basis());

		auto total_density = observables::density::total(spin_density);

		assert(total_density.basis() == vion_.basis());

		//IONIC POTENTIAL
		auto vscalar = vion_;

		//Time-dependent perturbation
		if(pert_.has_uniform_electric_field()){
			auto efield = pert_.uniform_electric_field(time);

			gpu::run(vscalar.basis().local_sizes()[2], vscalar.basis().local_sizes()[1], vscalar.basis().local_sizes()[0],
							 [point_op = vscalar.basis().point_op(), efield, _vscalar = vscalar.cubic().begin()] GPU_LAMBDA (auto iz, auto iy, auto ix){
								 auto rr = point_op.rvector_cartesian(ix, iy, iz);
								 _vscalar[ix][iy][iz] += -dot(efield, rr);
							 });
		}

		pert_.potential(time, vscalar);

		energy.external(operations::integral_product(total_density, vscalar));
		
		// Hartree
		if(theory_.hartree_potential()){
			auto vhartree = solvers::poisson::solve(total_density);
			energy.hartree(0.5*operations::integral_product(total_density, vhartree));
			operations::increment(vscalar, vhartree);
			if (spin_comp_charge.has_value()) {
				assert(spin_comp_charge.value().basis() == total_density.basis());
				auto total_nhat = observables::density::total(spin_comp_charge.value());
				auto vnhat = solvers::poisson::solve(total_nhat);
				operations::increment(vscalar, vnhat);
			}
		} else {
			energy.hartree(0.0);
		}

		auto vks = basis::field_set<basis::real_space, typename HamiltonianType::potential_type>(density_basis_, spin_density.set_size());

		gpu::run(spin_density.set_size(), density_basis_.local_size(), [_vscalar = vscalar.linear().cbegin(), _vks = vks.matrix().begin()] GPU_LAMBDA (auto ispin, auto ipoint) {
			if(ispin < 2) {
				_vks[ipoint][ispin] = _vscalar[ipoint];
			} else {
				_vks[ipoint][ispin] = 0.0;
			}
		});
		
		// XC
		double exc, nvxc;
		if(requires_kinetic_energy_density()) {
			assert(kinetic_energy_density.has_value());
			if(not hamiltonian.vmgga_.has_value()) hamiltonian.vmgga_.emplace(kinetic_energy_density->skeleton());
		}

		xc_(spin_density, core_density_, kinetic_energy_density, hamiltonian.vxc_, hamiltonian.vmgga_, exc, nvxc);

		//if the TDDFT functional is different from the ground state one we need to add a correction
		if(gs_vxc_.has_value()) {
			gpu::run(gs_vxc_->local_set_size(), gs_vxc_->basis().local_size(),
							 [_gs_vxc = gs_vxc_->matrix().cbegin(), _vxc = hamiltonian.vxc_.matrix().begin()] GPU_LAMBDA (auto is, auto ip){
								 _vxc[ip][is] += _gs_vxc[ip][is];
							 });

			exc  += gs_exc_;
			nvxc += gs_nvxc_;
		}
		
		assert(hamiltonian.vxc_.set_size() == vks.set_size());
		
		gpu::run(hamiltonian.vxc_.local_set_size(), hamiltonian.vxc_.basis().local_size(),
						 [_vxc = hamiltonian.vxc_.matrix().cbegin(), _vks = vks.matrix().begin()] GPU_LAMBDA (auto is, auto ip){
							 _vks[ip][is] += _vxc[ip][is];
						 });
		
		energy.xc(exc);
		energy.nvxc(nvxc);

		// PUT THE CALCULATED POTENTIAL IN THE HAMILTONIAN
		if(potential_basis_ == vks.basis()){
			hamiltonian.scalar_potential_= std::move(vks);
		} else {
			hamiltonian.scalar_potential_ = operations::transfer::coarsen(vks, potential_basis_);
		}

		// THE VECTOR POTENTIAL
		
		if(pert_.has_uniform_vector_potential()){
			hamiltonian.uniform_vector_potential_ = potential_basis_.cell().to_covariant(pert_.uniform_vector_potential(time));
		} else {
			hamiltonian.uniform_vector_potential_ = {0.0, 0.0, 0.0};
		}

		if(has_induced_vector_potential()){
			hamiltonian.uniform_vector_potential_ += induced_vector_potential_;
		}

		// THE MAGNETIC FIELD
		
		if (pert_.has_magnetic_field()) {
			basis::field<basis::real_space, vector3<double>> uniform_magnetic(spin_density.basis());
			uniform_magnetic.fill(vector3 {0.0, 0.0, 0.0});
			pert_.magnetic_field(time, uniform_magnetic);
			zeeman_coupling zc_(spin_density.set_size());
			auto zeeman_ener = 0.0;
			zc_(spin_density, uniform_magnetic, hamiltonian.scalar_potential_, zeeman_ener);
			energy.zeeman_energy(zeeman_ener);
		}
		
	}
	
	template <typename HamiltonianType, typename EnergyType, typename FieldType>
	void update_hamiltonian(HamiltonianType & hamiltonian, EnergyType & energy, FieldType const & spin_density, double time = 0.0) const {
		update_hamiltonian(hamiltonian, energy, spin_density, std::optional<FieldType>{}, std::optional<FieldType>{}, time);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename HamiltonianType, typename EnergyType, typename FieldType>
	void update_hamiltonian(HamiltonianType & hamiltonian, EnergyType & energy, FieldType const & spin_density,  std::optional<FieldType> const & kinetic_energy_density, double time = 0.0) const {
		update_hamiltonian(hamiltonian, energy, spin_density, kinetic_energy_density, std::optional<FieldType>{}, time);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Hamiltonian, typename Energy, typename Electrons>
	auto update_exchange(Hamiltonian & hamiltonian, Energy & energy, Electrons const & electrons) const {

		auto exxe = exact_exchange_.update(electrons, hamiltonian.ace_orbitals_, hamiltonian.ace_orbital_index_);
		energy.exact_exchange(exxe);
		
		return exxe;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////
	
	void propagate_induced_vector_potential(double const dt, vector3<double,covariant> const & current) {
		if(not has_induced_vector_potential()) return;
		solvers::velocity_verlet::propagate_positions(dt, theory_.alpha_value()*current/density_basis_.cell().volume(), induced_vector_potential_vel_, induced_vector_potential_);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	void propagate_induced_vector_potential_derivative(double const dt, vector3<double, covariant> const & current) {
		if(not has_induced_vector_potential()) return;
		solvers::velocity_verlet::propagate_velocities(dt, theory_.alpha_value()*current/density_basis_.cell().volume(), induced_vector_potential_vel_);
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto exx_coefficient() const {
		if(xc_.exchange().true_functional()) return xc_.exchange().exx_coefficient();
		return theory_.exchange_coefficient();
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	bool has_induced_vector_potential() const {
		return theory_.has_induced_vector_potential();
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	bool has_exact_exchange() const {
		return fabs(exx_coefficient()) > 1e-15;
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename HamiltonianType>
	void update_hamiltonian_nl(HamiltonianType & ham, systems::electrons & electrons){ 

		if (electrons.paw_option().has_value()) {

			auto vks = utils::field_set_to_vec(ham.scalar_potential(), 1);
			auto vxc = utils::field_set_to_vec(ham.vxc(), 1);

			electrons.paw().calculate_dij(vks, vxc);

			auto nprojs  = ham.projectors_all().nprojs();
			auto max_nlm = ham.projectors_all().max_nlm();
			gpu::array<double, 3> dij_full({nprojs, max_nlm, max_nlm}, 0.0);

			std::vector<double> dij_triangle;
			for (int iatom=0; iatom < nprojs; iatom++){
				auto nlm = ham.projectors_all().nlm()[iatom];
				auto size_upper_triangle = nlm * (nlm + 1) / 2;
				dij_triangle.resize(size_upper_triangle, 0);
				electrons.paw().get_dij(iatom, dij_triangle);
				int ind = -1;
				for (auto iproj = 0; iproj < nlm; iproj++){
					for (auto jproj = 0; jproj < iproj + 1; jproj++){
						ind = ind + 1;
						dij_full[iatom][jproj][iproj] = dij_triangle[ind];
						dij_full[iatom][iproj][jproj] = dij_full[iatom][jproj][iproj];
					}
				}
			}

			ham.projectors_all().update_nl_coeff(dij_full);

		}
	}
		
	
};
}
}
#endif

#ifdef INQ_HAMILTONIAN_SELF_CONSISTENCY_UNIT_TEST
#undef INQ_HAMILTONIAN_SELF_CONSISTENCY_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace Catch::literals;

}
#endif
