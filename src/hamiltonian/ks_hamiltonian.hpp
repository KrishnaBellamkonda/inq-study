/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__HAMILTONIAN__KS_HAMILTONIAN
#define INQ__HAMILTONIAN__KS_HAMILTONIAN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// ============================================================================
// PROPOSAL (muon per-orbital mass fork — 2026-07-06)
//
// WHY THIS FILE CHANGES:
//   This is the ROUTER between the per-state inverse mass (owned by `electrons`)
//   and the kinetic operator (`operations/laplacian.hpp`). Three call sites hard-
//   code the scalar -0.5 (= -hbar^2/2m, m_e=1):
//     :202  operations::laplacian(phi_fs, -0.5, ...)          (real-space apply)
//     :235  operations::laplacian_add(phi, hphi, -0.5, ...)   (fourier apply)
//     :245  operations::laplacian_expectation_value(phi, -0.5,...) (ENERGY ledger)
//   Each must instead pass a PER-STATE factor array = -0.5 * inverse_mass[ist],
//   partition-aligned to `phi`. Fixing :245 keeps E_total consistent for free
//   (energy.hpp weights this by occupations externally).
//
// HOW IT FITS THE BROADER SCOPE:
//   Holds the per-state inverse mass as a member, set at construction from
//   electrons.inverse_mass(). Constructor argument DEFAULTS to "all ones" so the
//   ground_state construction sites (calculator.hpp, initial_guess.hpp) need NO
//   change and GS stays bit-for-bit identical — only real_time/propagate.hpp
//   opts in by passing the real electrons.inverse_mass().
//
// HOW THE WORKFLOW IS AFFECTED:
//   None at the user level; this is pure engine plumbing behind `inq-run`.
//
// DATAFLOW (per-state mass — the full call chain, starting from run.cpp):
//   1. run.cpp                    electrons.inverse_mass()[0][wp_idx] = 1.0/M_MU   (set the knob)
//   2. real_time/propagate.hpp    ham.set_inverse_mass(electrons.inverse_mass()[0])
//   3. ks_hamiltonian::set_inverse_mass  -> stores kinetic_factor_[ist] = -0.5*im[ist]
//                                           (left EMPTY when every mass == 1 -> scalar path)
//   4. ks_hamiltonian apply / kinetic_expectation_value  branch on kinetic_factor_.size():
//        empty -> operations::laplacian / laplacian_add / laplacian_expectation_value   (scalar -0.5)
//        set   -> operations::laplacian_states / laplacian_add_states /
//                 laplacian_expectation_value_states   (per-state factor; operations/laplacian.hpp)
//   5. hamiltonian/energy.hpp sums the (occupation-weighted) kinetic expectation into E_total.
//
//   Operator placement: the kinetic Laplacian is applied in FOURIER space (phi is
//   transformed to reciprocal space first); the local/Hartree/xc potentials are
//   applied in REAL space via scalar_potential_add. The mass fork touches only the
//   kinetic (Laplacian) path.
// ============================================================================

#include <basis/field.hpp>
#include <multi/adaptors/fftw.hpp>
#include <hamiltonian/atomic_potential.hpp>
#include <hamiltonian/exchange_operator.hpp>
#include <hamiltonian/projector.hpp>
#include <hamiltonian/projector_all.hpp>
#include <hamiltonian/relativistic_projector.hpp>
#include <hamiltonian/scalar_potential.hpp>
#include <input/environment.hpp>
#include <operations/transform.hpp>
#include <operations/divergence.hpp>
#include <operations/laplacian.hpp>
#include <operations/gradient.hpp>
#include <states/index.hpp>
#include <states/ks_states.hpp>
#include <states/orbital_set.hpp>

#include <utils/profiling.hpp>

#include <list>
#include <unordered_map>

namespace inq {
namespace hamiltonian {

template <typename PotentialType>
class ks_hamiltonian {
		
public:

	using potential_type = PotentialType;

private:

	basis::field_set<basis::real_space, double> vxc_;
	basis::field_set<basis::real_space, PotentialType> scalar_potential_;
	std::optional<basis::field_set<basis::real_space, double>> vmgga_;
	vector3<double, covariant> uniform_vector_potential_;
	projector_all projectors_all_;		
	std::list<relativistic_projector> projectors_rel_;
	states::ks_states states_;
	std::vector<states::orbital_set<basis::real_space, complex>> ace_orbitals_;
	states::index ace_orbital_index_;
	// Per-state kinetic prefactor (= -0.5 * inverse_mass[ist]), local + partition-
	// aligned to phi (muon mass fork, 2026-07-06). EMPTY by default → the kinetic
	// operator uses the ORIGINAL scalar -0.5 path (bit-for-bit identical to the
	// unforked engine). Populated via set_inverse_mass() ONLY when some mass != 1.
	gpu::array<double, 1> kinetic_factor_;

#ifdef ENABLE_CUDA
public:
#endif
		
		template <typename OccType, typename ArrayType>
		static double occ_sum(OccType const & occupations, ArrayType const & array) {
			CALI_CXX_MARK_FUNCTION;
			
			assert(occupations.size() == array.size());
			return gpu::run(gpu::reduce(array.size()), 0.0, [_occupations = occupations.cbegin(), _array = array.cbegin()] GPU_LAMBDA (auto ip) {
				return _occupations[ip]*real(_array[ip]);
			});
		}

public:
	
	void update_projectors(const basis::real_space & basis, const atomic_potential & pot, systems::ions const & ions){
			
		CALI_CXX_MARK_FUNCTION;

		std::list<projector> projectors;
			
		for(int iatom = 0; iatom < ions.size(); iatom++){
			auto && ps = pot.pseudo_for_element(ions.species(iatom));

			if(ps.has_total_angular_momentum()){
				projectors_rel_.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors_rel_.back().empty()) projectors_rel_.pop_back();
			} else {
				projectors.emplace_back(basis, pot.double_grid(), ps, ions.positions()[iatom], iatom);
				if(projectors.back().empty()) projectors.pop_back();
			}
		}

		projectors_all_ = projector_all(projectors);
	}
		

	////////////////////////////////////////////////////////////////////////////////////////////

	// Constructor is UNCHANGED (every existing caller is source-compatible). The
	// per-state mass is opted into afterwards via set_inverse_mass() — see below.
	ks_hamiltonian(const basis::real_space & basis, states::ks_states const & states, atomic_potential const & pot, systems::ions const & ions):
		vxc_(basis, states.num_density_components()),
		scalar_potential_(basis, states.num_density_components()),
		uniform_vector_potential_({0.0, 0.0, 0.0}),
		states_(states)
	{
		scalar_potential_.fill(0.0);
		update_projectors(basis, pot, ions);
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	// Opt into a per-state mass (muon mass fork, 2026-07-06). `im` is the LOCAL
	// per-state inverse-mass array for this kpin (e.g. electrons.inverse_mass()[0]).
	// If every entry is exactly 1.0, kinetic_factor_ is left EMPTY so the kinetic
	// operator keeps the original scalar -0.5 path (bit-for-bit unchanged). Only
	// when some mass != 1 do we build kinetic_factor_[ist] = -0.5 * im[ist].
	template <class InvMassArray>
	void set_inverse_mass(InvMassArray const & im){
		CALI_CXX_MARK_FUNCTION;
		auto const dev = gpu::run(gpu::reduce(im.size()), 0.0,
			[imit = begin(im)] GPU_LAMBDA (auto i){ return fabs(imit[i] - 1.0); });
		if(dev == 0.0){                 // trivial (all electrons) → keep scalar path
			kinetic_factor_ = gpu::array<double, 1>{};
			return;
		}
		kinetic_factor_ = gpu::array<double, 1>(im.size());
		gpu::run(im.size(),
			[kf = begin(kinetic_factor_), imit = begin(im)] GPU_LAMBDA (auto i){ kf[i] = -0.5*imit[i]; });
	}

	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto non_local(const states::orbital_set<basis::real_space, complex> & phi) const {

		CALI_CXX_MARK_FUNCTION;
 
		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);
		
		states::orbital_set<basis::real_space, complex> vnlphi(phi.skeleton());
		vnlphi.fill(0.0);
		
		projectors_all_.apply(proj, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		for(auto & pr : projectors_rel_) pr.apply(phi, vnlphi, phi.kpoint() + uniform_vector_potential_);
		
		return vnlphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <typename Occupations>
	auto non_local_energy(states::orbital_set<basis::real_space, complex> const & phi, Occupations const & occupations, bool const reduce_states = true) const {

		CALI_CXX_MARK_FUNCTION;

		auto en = projectors_all_.energy(phi, phi.kpoint() + uniform_vector_potential_, occupations, reduce_states);
		for(auto & pr : projectors_rel_) en += pr.energy(phi, occupations, phi.kpoint() + uniform_vector_potential_);
		return en;
		
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	void mgga_term(states::orbital_set<basis::real_space, complex> const & phi, states::orbital_set<basis::real_space, complex> & hphi) const {
		//THIS PROBABLY CAN BE DONE MORE EFFICIENTLY BY REUSING THE FFTS IN THE HAMILTONIAN
		CALI_CXX_MARK_FUNCTION;
		
		if(vmgga_.has_value()) {
			auto gphi = operations::gradient(phi);
			gpu::run(gphi.local_set_size(), gphi.basis().local_size(),
							 [ispin = phi.spin_index(), _gphi = begin(gphi.matrix()), _vmgga = begin(vmgga_->matrix())] GPU_LAMBDA (auto ist, auto ip) {
								 _gphi[ip][ist] *= _vmgga[ip][ispin];
							 });
			auto div = operations::divergence(gphi);
			gpu::run(hphi.local_set_size(), hphi.basis().local_size(),
							 [_hphi = begin(hphi.matrix()), _div = begin(div.matrix())] GPU_LAMBDA (auto ist, auto ip) {
								 _hphi[ip][ist] -= 0.5*_div[ip][ist];
							 });
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	template <typename Occupations>
	double mgga_energy(states::orbital_set<basis::real_space, complex> const & phi, Occupations const & occupations, bool const reduce_states = true) const {
		if(not vmgga_.has_value()) return 0.0;

		CALI_CXX_MARK_FUNCTION;
		
		auto gphi = operations::gradient(phi);
		auto en = gpu::run(gpu::reduce(gphi.local_set_size()), gpu::reduce(gphi.basis().local_size()), 0.0,
											 [ispin = phi.spin_index(), _gphi = begin(gphi.matrix()), _vmgga = begin(vmgga_->matrix()), _occupations = begin(occupations), cell = phi.basis().cell()] GPU_LAMBDA (auto ist, auto ip) {
												 return _occupations[ist]*_vmgga[ip][ispin]*cell.norm(_gphi[ip][ist]);
											 });

		assert(reduce_states == false);

		return phi.basis().volume_element()*en;
	}

	//////////////////////////////////////////////////////////////////////////////////
	
	void ace(const states::orbital_set<basis::real_space, complex> & phi, states::orbital_set<basis::real_space, complex> & exxphi) const {
		if(ace_orbitals_.size() == 0) return;
		
		CALI_CXX_MARK_SCOPE("hamiltonian::ace");
		namespace blas = boost::multi::blas;
		
		auto index = ace_orbital_index_.at(phi.key());
		
		assert(phi.kpoint() == ace_orbitals_[index].kpoint());
		assert(phi.spin_index() == ace_orbitals_[index].spin_index());
		
		auto olap = operations::overlap(ace_orbitals_[index], phi);
		operations::rotate(olap, ace_orbitals_[index], exxphi, -1.0, 1.0);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("hamiltonian_real");

		auto proj = projectors_all_.project(phi, phi.kpoint() + uniform_vector_potential_);

		auto phi_fs = operations::transform::to_fourier(phi);
		
		// Kinetic term. Empty kinetic_factor_ → original scalar -0.5 (electrons).
		// Non-empty → per-state factor: electrons keep -0.5, the muon slot -0.5/m_mu.
		auto const gradcoeff = -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_);
		auto hphi_fs = (kinetic_factor_.size() == 0)
			? operations::laplacian(phi_fs, -0.5, gradcoeff)
			: operations::laplacian_states(phi_fs, kinetic_factor_, gradcoeff);
			
		auto hphi = operations::transform::to_real(hphi_fs);

		hamiltonian::scalar_potential_add(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().norm(phi.kpoint() + uniform_vector_potential_), phi, hphi);
		ace(phi, hphi);
		mgga_term(phi, hphi);
		
		for(auto & pr : projectors_rel_) pr.apply(phi, hphi, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi, phi.kpoint() + uniform_vector_potential_);

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto operator()(const states::orbital_set<basis::fourier_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("hamiltonian_fourier");

		auto phi_rs = operations::transform::to_real(phi);

		auto proj = projectors_all_.project(phi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi_rs = hamiltonian::scalar_potential(scalar_potential_, phi.spin_index(), 0.5*phi.basis().cell().norm(phi.kpoint() + uniform_vector_potential_), phi_rs);
		ace(phi_rs, hphi_rs);
		mgga_term(phi_rs, hphi_rs);

		for(auto & pr : projectors_rel_) pr.apply(phi_rs, hphi_rs, phi.kpoint() + uniform_vector_potential_);
		projectors_all_.apply(proj, hphi_rs, phi.kpoint() + uniform_vector_potential_);
			
		auto hphi = operations::transform::to_fourier(hphi_rs);

		// Kinetic term (fourier-space entry point) — same guard as the real-space apply.
		auto const gradcoeff = -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_);
		if(kinetic_factor_.size() == 0) operations::laplacian_add(phi, hphi, -0.5, gradcoeff);
		else                            operations::laplacian_add_states(phi, hphi, kinetic_factor_, gradcoeff);

		return hphi;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto kinetic_expectation_value(states::orbital_set<basis::fourier_space, complex> const & phi) const {
		CALI_CXX_MARK_FUNCTION;

		// CRITICAL for the energy ledger — SAME guard/factor as the apply kernels.
		// energy.hpp sums this (occupation-weighted) into E_kinetic; a mismatch vs
		// the apply kernels would silently corrupt E_total for the muon.
		auto const gradcoeff = -2.0*phi.basis().cell().to_contravariant(phi.kpoint() + uniform_vector_potential_);
		return (kinetic_factor_.size() == 0)
			? operations::laplacian_expectation_value(phi, -0.5, gradcoeff)
			: operations::laplacian_expectation_value_states(phi, kinetic_factor_, gradcoeff);
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto kinetic_expectation_value(states::orbital_set<basis::real_space, complex> const & phi) const {
		CALI_CXX_MARK_FUNCTION;

		auto phi_fs = operations::transform::to_fourier(phi);
		return kinetic_expectation_value(phi_fs);
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	auto has_overlap() const {
		return projectors_all_.has_overlap();
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////

	struct overlap_functor {
		projector_all const & proj_all_;
		vector3<double, covariant> vector_potential_;

		auto operator()(const states::orbital_set<basis::real_space, complex> & phi) const {
			
			CALI_CXX_MARK_SCOPE("overlap_real");
			
			auto proj = proj_all_.project_overlap(phi, phi.kpoint() + vector_potential_);
			
			auto sphi = phi;
			proj_all_.apply(proj, sphi, phi.kpoint() + vector_potential_);
			
			return sphi;
		}
	};

	////////////////////////////////////////////////////////////////////////////////////////////

	auto overlap_operator() const {
		return overlap_functor{projectors_all_, uniform_vector_potential_};
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto overlap(const states::orbital_set<basis::fourier_space, complex> & phi) const {
			
		CALI_CXX_MARK_SCOPE("overlap_real");

		auto phi_rs = operations::transform::to_real(phi);
		auto sphi_rs = overlap_operator()(phi_rs);
		auto sphi_fs = operations::transform::to_fourier(sphi_rs);

		return sphi_fs;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto momentum(const states::orbital_set<basis::real_space, complex> & phi) const{
		CALI_CXX_MARK_FUNCTION;

		return operations::gradient(phi, /* factor = */ 1.0, /*shift = */ phi.kpoint() + uniform_vector_potential_);
	}
		
	////////////////////////////////////////////////////////////////////////////////////////////
		
	auto & projectors_all() const {
		return projectors_all_;
	}

	auto & projectors_all() {
		return projectors_all_;
	}
	////////////////////////////////////////////////////////////////////////////////////////////
	
	auto & projectors_rel() const {
		return projectors_rel_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	template <class output_stream>
	void info(output_stream & out) const {
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & scalar_potential() {
		return scalar_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & uniform_vector_potential() const {
		return uniform_vector_potential_;
	}
	auto & uniform_vector_potential() {
		return uniform_vector_potential_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////

	auto & vxc() const {
		return vxc_;
	}

	auto & vxc() {
		return vxc_;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename Perturbation>
	friend class self_consistency;
	
};
}
}
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
#endif

#ifdef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST
#undef INQ_HAMILTONIAN_KS_HAMILTONIAN_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>
#include <config/path.hpp>
TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG){

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Hamiltonian application") {
		
		parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});
		
		auto set_comm = basis::set_subcomm(cart_comm);
		auto basis_comm = basis::basis_subcomm(cart_comm);
		
		auto ions = systems::ions(systems::cell::cubic(10.0_b));

		basis::real_space rs(ions.cell(), /*spacing = */ 0.49672941, basis_comm);
		
		CHECK(rs.size() == 8000);
		CHECK(rs.rspacing()[0] == 0.5_a);
		CHECK(rs.rspacing()[1] == 0.5_a);	
		CHECK(rs.rspacing()[2] == 0.5_a);
		CHECK(rs.volume_element() == 0.125_a);
	
		hamiltonian::atomic_potential pot(ions.species_list(), rs.gcutoff());
		
		states::ks_states st(states::spin_config::UNPOLARIZED, 11.0);
		
		states::orbital_set<basis::real_space, complex> phi(rs, st.num_states(), 1, vector3<double, covariant>{0.0, 0.0, 0.0}, 0, cart_comm);
		
		hamiltonian::ks_hamiltonian<double> ham(rs, st, pot, ions);
		
		auto const nst = phi.local_set_size();

		//Constant function
		{
			ham.scalar_potential().fill(0);
			phi.fill(1.0);

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));

			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [nst, _hphi_r = begin(hphi_r.hypercubic()), _hphi_f = begin(hphi_f.hypercubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {
													 
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++) {
															 acc_r += fabs(_hphi_r[ix][iy][iz][ist] - 0.0);
															 acc_f += fabs(_hphi_f[ix][iy][iz][ist] - 0.0);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();
		
			CHECK(real(diff) < 1e-14);
			CHECK(imag(diff) < 1e-14);
		
		}
		
		//Plane wave
		{
		
			double kk = 2.0*M_PI/rs.rlength()[0];
		
			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [kk, nst, pot = begin(ham.scalar_potential().hypercubic()), _phi = begin(phi.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
								 pot[ix][iy][iz][0] = 0.0;

								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
					
								 for(int ist = 0; ist < nst; ist++){
								 
									 auto istg = set_part.local_to_global(ist);
									 double xx = point_op.rvector_cartesian(ixg, iyg, izg)[0];
									 _phi[ix][iy][iz][ist] = complex(cos(istg.value()*kk*xx), sin(istg.value()*kk*xx));
								 }
							 });

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [kk, nst, _phi = begin(phi.hypercubic()), _hphi_r = begin(hphi_r.hypercubic()), _hphi_f = begin(hphi_f.hypercubic()), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {
													 
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++){
															 auto istg = set_part.local_to_global(ist);
															 acc_r += fabs(_hphi_r[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*_phi[ix][iy][iz][ist]);
															 acc_f += fabs(_hphi_f[ix][iy][iz][ist] - 0.5*istg.value()*kk*istg.value()*kk*_phi[ix][iy][iz][ist]);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();

			CHECK(real(diff) < 1e-14);
			CHECK(imag(diff) < 1e-14);
		
		}

		//Harmonic oscillator
		{
			double ww = 2.0;

			gpu::run(rs.local_sizes()[2], rs.local_sizes()[1], rs.local_sizes()[0],
							 [ww, nst, pot = begin(ham.scalar_potential().hypercubic()), _phi = begin(phi.hypercubic()), point_op = rs.point_op(),
								part0 = rs.cubic_part(0), part1 = rs.cubic_part(1), part2 = rs.cubic_part(2), set_part = phi.set_part()] GPU_LAMBDA (auto iz, auto iy, auto ix) {

								 auto ixg = part0.local_to_global(ix);
								 auto iyg = part1.local_to_global(iy);
								 auto izg = part2.local_to_global(iz);
							 
								 double r2 = point_op.r2(ixg, iyg, izg);
								 pot[ix][iy][iz][0] = 0.5*ww*ww*r2;
							 
								 for(int ist = 0; ist < nst; ist++) _phi[ix][iy][iz][ist] = exp(-ww*r2);
							 });

			auto hphi_r = ham(phi);
			auto hphi_f = operations::transform::to_real(ham(operations::transform::to_fourier(phi)));
		
			auto diff = gpu::run(gpu::reduce(rs.local_sizes()[2]), gpu::reduce(rs.local_sizes()[1]), gpu::reduce(rs.local_sizes()[0]), complex{0.0, 0.0},
													 [ww, nst, _phi = begin(phi.hypercubic()), _hphi_r = begin(hphi_r.hypercubic()), _hphi_f = begin(hphi_f.hypercubic())] GPU_LAMBDA (auto iz, auto iy, auto ix) {
														 auto acc_r = 0.0;
														 auto acc_f = 0.0;
														 for(int ist = 0; ist < nst; ist++)	{
															 acc_r += fabs(_hphi_r[ix][iy][iz][ist] - 1.5*ww*_phi[ix][iy][iz][ist]);
															 acc_f += fabs(_hphi_f[ix][iy][iz][ist] - 1.5*ww*_phi[ix][iy][iz][ist]);
														 }
														 return complex{acc_r, acc_f};
													 });
		
			cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
			diff /= phi.set_size()*phi.basis().size();
		
			CHECK(real(diff) == 0.0051420503_a);
			CHECK(imag(diff) == 0.0051420503_a);
		}
	}
	
	SECTION("PAW Overlap operator"){

		auto cell = systems::cell::cubic(30.0_b).finite();
		systems::ions ions(cell);
		ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {0.0_b, 0.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {2.0_b, 0.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {0.0_b, 2.0_b, 0.0_b});
		ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {0.0_b, 0.0_b, 2.0_b});
		ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {0.0_b, 0.0_b, -10.0_b});
		systems::electrons electrons(ions, input::kpoints::gamma(), options::electrons{}.cutoff(300.0_Ha).extra_states(0));

		hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.states(), electrons.atomic_pot(), ions);

		auto overlap_operator = ham.overlap_operator();

		for(auto & phi : electrons.kpin()) {
			phi.fill(1.0);

			auto sphi = overlap_operator(phi);
			operations::shift(-1.0, sphi, phi);
			auto olap = operations::overlap(phi);
			auto olap_array = matrix::all_gather(olap);

			CHECK(real(olap_array[0][0]) == Approx(0.97031883));
			CHECK(imag(olap_array[0][0]) == 0.0);
		}
	}
	}
#endif
