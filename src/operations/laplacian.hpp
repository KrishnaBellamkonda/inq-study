/* -*- indent-tabs-mode: t -*- */

#ifndef OPERATIONS__LAPLACIAN
#define OPERATIONS__LAPLACIAN

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>

#include <gpu/run.hpp>
#include <basis/field_set.hpp>
#include <basis/fourier_space.hpp>
#include <math/vector3.hpp>
#include <operations/transform.hpp>
#include <utils/profiling.hpp>

// ============================================================================
// PROPOSAL (muon per-orbital mass fork — 2026-07-06)
//
// WHY THIS FILE CHANGES:
//   This is where -hbar^2/2m is actually applied. Every kernel below multiplies
//   the (-g2 + gradcoeff.gvec) laplacian by a SINGLE SCALAR `factor` (= -0.5)
//   for ALL states `ist`. The muon needs that prefactor PER STATE:
//   -0.5*inverse_mass[ist] (i.e. -0.5 for electrons, -0.5/m_mu for the WP slot).
//
// HOW IT FITS THE BROADER SCOPE:
//   COMPUTE layer of the fork. ks_hamiltonian builds a per-state factor array
//   (length = ff.local_set_size(), already partition-aligned) from
//   electrons.inverse_mass() and passes it here instead of the scalar -0.5.
//   gradcoeff and the folded k2 term sit INSIDE the same `factor*(...)` product,
//   so per-state scaling scales the whole 1/(2m)(-i grad + A)^2 consistently —
//   EXCEPT the separate +A^2/2m term added via scalar_potential_add in
//   ks_hamiltonian, which is zero at gamma / zero vector potential (the WP muon
//   regime). A vector-potential kick would need extra work (documented).
//
// HOW THE WORKFLOW IS AFFECTED:
//   Implement as OVERLOADS taking a per-state factor array so the scalar API
//   used everywhere else stays bit-for-bit identical. Only three functions are
//   on the muon path: laplacian_add, laplacian, laplacian_expectation_value.
//   (laplacian_in_place below is NOT used by ks_hamiltonian — leave it alone.)
//
// MY THOUGHTS:
// 	 This is the file where the actual mass revision is applied. Here, the laplacian
//   is used to calculate the kinteic energy. So the laplacian operator is applied here using emplace if I right. So, we apply it here.
//   If my understanding is right, we need to make a inverse_mass array and multiply each of the states
//   ist here by the inverse mass before the Laplcaian is applied. I don't quite understand what other
//   changes are made here. I think most of them are about adding a pre-factor, but I don't understand
//   what specific purpose this satisfies. 
//
// ============================================================================

namespace inq {
namespace operations {

// PROPOSAL: add a per-state-factor overload of laplacian_add; kernel becomes
//   auto lapl = factor_arr[ist]*(-g2 + dot(gradcoeff, gvec));
// Keep this scalar version for all non-muon callers.
template <typename FieldSetType, typename FactorType = double>
void laplacian_add(FieldSetType const & ff, FieldSetType & laplff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");
			
	gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), factor, gradcoeff]
					 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 laplffcub[i0][i1][i2][ist] += lapl*ffcub[i0][i1][i2][ist];
					 });
	
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorType = double>
void laplacian_in_place(FieldSetType & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");

	gpu::run(ff.set_part().local_size(), ff.basis().local_sizes()[2], ff.basis().local_sizes()[1], ff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(),
						ffcub = begin(ff.hypercubic()), factor, gradcoeff] GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 ffcub[i0][i1][i2][ist] = ffcub[i0][i1][i2][ist]*lapl;
					 });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// PROPOSAL: add a per-state-factor overload of laplacian (real- + fourier-space
// branches). This is the version called at ks_hamiltonian.hpp:202. Kernel:
//   auto lapl = factor_arr[ist]*(-g2 + dot(gradcoeff, gvec));
template <typename FieldSetType, typename FactorType = double>
FieldSetType laplacian(FieldSetType const & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	if constexpr(std::is_same_v<typename FieldSetType::basis_type, basis::real_space>) {
		if constexpr(std::is_same_v<typename FieldSetType::element_type, double>) {
			return real_field(operations::transform::to_real(operations::laplacian(operations::transform::to_fourier(complex_field(ff)), factor)));
		} else {
			return operations::transform::to_real(operations::laplacian(operations::transform::to_fourier(ff), factor));
		}
	} else {
		
		static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for real or fourier_space");
		
		FieldSetType laplff(ff.skeleton());
		
		gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
						 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), factor, gradcoeff]
						 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
							 auto lapl = factor*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
							 laplffcub[i0][i1][i2][ist] = lapl*ffcub[i0][i1][i2][ist];
						 });
		
		return laplff;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// PROPOSAL: add a per-state-factor overload of laplacian_expectation_value —
// this is the ENERGY-LEDGER path (called at ks_hamiltonian.hpp:245, summed in
// hamiltonian/energy.hpp:82). It MUST use the SAME per-state factor as the apply
// kernels, or the muon's kinetic energy in E_total is silently wrong. Kernel:
//   auto lapl = fac_arr[ist]*(-g2 + dot(gradcoeff, gvec) + k2);   // fac_arr[ist] = factor_arr[ist]*volume_element
template <typename FieldSetType, typename FactorType = double>
gpu::array<double, 1> laplacian_expectation_value(FieldSetType & ff, FactorType factor = 1.0, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");

	
	auto k2 = -0.25*ff.basis().cell().dot(gradcoeff, gradcoeff);
	
	auto evs = gpu::run(ff.local_set_size(), gpu::reduce(ff.basis().local_sizes()[2]), gpu::reduce(ff.basis().local_sizes()[1]), gpu::reduce(ff.basis().local_sizes()[0]), 0.0,
											[point_op = ff.basis().point_op(), ffcub = begin(ff.hypercubic()), fac = factor*ff.basis().volume_element(), gradcoeff, k2]
											GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
												auto lapl = fac*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)) + k2);
												return real(conj(ffcub[i0][i1][i2][ist])*lapl*ffcub[i0][i1][i2][ist]);
											});
	
	if(ff.spinor_dim() == 2) {
		auto spinor_evs = decltype(evs)(ff.local_spinor_set_size());
		gpu::run(ff.local_spinor_set_size(), [sev = begin(spinor_evs), ev = begin(evs), nst = ff.local_spinor_set_size()] GPU_LAMBDA (auto ist) {
			sev[ist] = ev[ist] + ev[ist + nst];
		});
		
		evs = std::move(spinor_evs);
	}

	if(ff.basis().comm().size() > 1) {
		ff.basis().comm().all_reduce_in_place_n(raw_pointer_cast(evs.data_elements()), evs.num_elements(), std::plus<>{});
	}
	
	return evs;

}

///////////////////////////////////////////////////////////////////////////////////////////////////
// ---- per-state factor variants (muon mass fork, 2026-07-06) --------------------
// `factors[ist]` is the per-state kinetic prefactor (= -0.5 * inverse_mass[ist]).
// Distinct names (NOT overloads) so the scalar API above is untouched → the
// Tier-0 "inert when off" bit-for-bit invariant is protected. Fourier-space only
// (the callers in ks_hamiltonian pass the fourier transform).

template <typename FieldSetType, typename FactorsArray>
void laplacian_add_states(FieldSetType const & ff, FieldSetType & laplff, FactorsArray const & factors, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");
	assert(long(factors.size()) == laplff.set_part().local_size());

	gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), fac = begin(factors), gradcoeff]
					 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = fac[ist]*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 laplffcub[i0][i1][i2][ist] += lapl*ffcub[i0][i1][i2][ist];
					 });
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorsArray>
FieldSetType laplacian_states(FieldSetType const & ff, FactorsArray const & factors, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");
	assert(long(factors.size()) == ff.set_part().local_size());

	FieldSetType laplff(ff.skeleton());

	gpu::run(laplff.set_part().local_size(), laplff.basis().local_sizes()[2], laplff.basis().local_sizes()[1], laplff.basis().local_sizes()[0],
					 [point_op = ff.basis().point_op(), laplffcub = begin(laplff.hypercubic()), ffcub = begin(ff.hypercubic()), fac = begin(factors), gradcoeff]
					 GPU_LAMBDA (auto ist, auto i2, auto i1, auto i0){
						 auto lapl = fac[ist]*(-point_op.g2(i0, i1, i2) + dot(gradcoeff, point_op.gvector(i0, i1, i2)));
						 laplffcub[i0][i1][i2][ist] = lapl*ffcub[i0][i1][i2][ist];
					 });

	return laplff;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename FieldSetType, typename FactorsArray>
gpu::array<double, 1> laplacian_expectation_value_states(FieldSetType & ff, FactorsArray const & factors, vector3<double, contravariant> const & gradcoeff = {0.0, 0.0, 0.0}){

	CALI_CXX_MARK_FUNCTION;

	static_assert(std::is_same_v<typename FieldSetType::basis_type, basis::fourier_space>, "Only implemented for fourier_space");

	// The kinetic energy per state is LINEAR in its prefactor:
	//   <psi_i| factor_i*(-g^2+..) |psi_i> = (factor_i / -0.5) * <psi_i| -0.5*(-g^2+..) |psi_i>.
	// So reuse the GPU-proven SCALAR reduce (factor = -0.5) and then rescale each
	// per-state result by (factor_i / -0.5) = -2*factor_i with a PLAIN element-wise
	// kernel. This deliberately AVOIDS a custom reduce lambda that captures a device
	// array-iterator — nvcc forbids querying such an extended-__device__ lambda's
	// return type in host code (the CUB 4-D reduce path does exactly that), which
	// broke the whole engine's GPU build. factors[i] = -0.5*inverse_mass[i], so the
	// rescale factor -2*factors[i] = inverse_mass[i] — i.e. scale electron KE by 1/m.
	auto evs = laplacian_expectation_value(ff, -0.5, gradcoeff);   // per-state, GPU-proven
	assert(long(factors.size()) == long(evs.size()));
	gpu::run(evs.size(), [ev = begin(evs), fac = begin(factors)] GPU_LAMBDA (auto i){
		ev[i] = -2.0*fac[i]*ev[i];
	});
	return evs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


}
}
#endif

#ifdef INQ_OPERATIONS_LAPLACIAN_UNIT_TEST
#undef INQ_OPERATIONS_LAPLACIAN_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace inq::magnitude;	
	using namespace Catch::literals;
	using namespace operations;
	using Catch::Approx;
	
	parallel::cartesian_communicator<2> cart_comm(boost::mpi3::environment::get_world_instance(), {});
	auto set_comm = basis::set_subcomm(cart_comm);
	auto basis_comm = basis::basis_subcomm(cart_comm);	

	SECTION("Plane-wave -- field_set"){
		
		double lx = 9;
		double ly = 12;
		double lz = 10;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::orthorhombic(lx*1.0_b, ly*1.0_b, lz*1.0_b), /*spacing =*/ 0.49672941, basis_comm);

		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		//Define k-vector for test function
		auto kvec = 2.0*M_PI*vector3<double>(1.0/lx, 1.0/ly, 1.0/lz);
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-8) ;
		CHECK(diff_in_place < 1.0e-8);
		CHECK(diff_add < 5.0e-7);

	}

	SECTION("Plane-wave -- per-state mass (muon fork)"){
		// Tier-1 (T1.1/T1.3/T1.4): the per-state apply kernels must give each state
		// its own kinetic prefactor factor[ist] = -0.5*inverse_mass[ist]. Three
		// electron states (m=1 -> -0.5) plus one "muon" state (m=206.77 -> -0.5/m).
		// Oracle: laplacian_states result_ist = factor[ist]*(-|k|^2)*psi.

		double lx = 9, ly = 12, lz = 10;
		double const m_mu = 206.77;
		int const nst = 4;

		basis::real_space rs(systems::cell::orthorhombic(lx*1.0_b, ly*1.0_b, lz*1.0_b), /*spacing =*/ 0.49672941, basis_comm);
		basis::field_set<basis::real_space, complex> func(rs, nst, cart_comm);

		auto kvec = 2.0*M_PI*vector3<double>(1.0/lx, 1.0/ly, 1.0/lz);
		auto ff = [] (auto & kk, auto & rr){ return exp(inq::complex(0.0, 1.0)*dot(kk, rr)); };

		for(int ix = 0; ix < rs.local_sizes()[0]; ix++)
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++)
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = ff(kvec, vec);
				}

		// per-state factor array; muon = last GLOBAL state (index nst-1)
		gpu::array<double, 1> factors(func.local_set_size());
		auto inv_mass_of = [&](long istg){ return (istg == nst - 1) ? 1.0/m_mu : 1.0; };
		for(int ist = 0; ist < func.local_set_size(); ist++){
			auto istg = func.set_part().start() + ist;
			factors[ist] = -0.5*inv_mass_of(istg);
		}

		auto func_fs = operations::transform::to_fourier(func);
		auto lapl = operations::transform::to_real(operations::laplacian_states(func_fs, factors));

		// in-place-add variant (fourier entry point used at ks_ham :235): start from
		// one kinetic term (laplacian_states) then add a second → expect 2x.
		auto acc_fs = operations::laplacian_states(operations::transform::to_fourier(func), factors);
		operations::laplacian_add_states(operations::transform::to_fourier(func), acc_fs, factors);
		auto lapl_add = operations::transform::to_real(acc_fs);

		double diff = 0.0, diff_add = 0.0;
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++)
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++)
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto istg = func.set_part().start() + ist;
						auto factor = -0.5*inv_mass_of(istg);
						auto anvalue = factor*(-norm(kvec))*ff(kvec, vec);   // factor*(-|k|^2)*psi
						diff     += fabs(lapl.hypercubic()[ix][iy][iz][ist]     - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue); // one term + one added = 2x
					}
				}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});

		CHECK(diff < 1.0e-8);           // laplacian_states: per-state factor applied exactly
		CHECK(diff_add < 5.0e-7);       // laplacian_add_states: func_fs2 held one kinetic term, add gives 2x
	}

	SECTION("Plane-wave -- rotated"){
		
		double ll = 9;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::lattice({ll*1.0_b/sqrt(2), ll*1.0_b/sqrt(2), 0.0_b}, {-ll*1.0_b/sqrt(2), ll*1.0_b/sqrt(2), 0.0_b}, {0.0_b, 0.0_b, ll*1.0_b}), /*spacing =*/ 0.49672941, basis_comm);
		
		CHECK(rs.cell().volume() == ll*ll*ll);
		
		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		auto kvec = 2.0*M_PI*vector3<double>(sqrt(2.0)/ll, sqrt(2.0)/ll, 0.0);
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};

		CHECK(norm(kvec) == Approx(rs.cell().norm(rs.cell().to_covariant(kvec))));
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-8) ;
		CHECK(diff_in_place < 1.0e-8);
		CHECK(diff_add < 5.0e-7);
		
	}
	
	SECTION("Plane-wave -- non-orthogonal"){
		
		double ll = 5.89;
		double factor = 0.673214;
	
		basis::real_space rs(systems::cell::lattice({0.0_b, ll*1.0_b, ll*1.0_b}, {ll*1.0_b, 0.0_b, ll*1.0_b}, {ll*1.0_b, ll*1.0_b, 0.0_b}), /*spacing =*/ 0.49672941, basis_comm);
		
		basis::field_set<basis::real_space, complex> func(rs, 13, cart_comm);
	
		auto kvec = rs.cell().to_cartesian(2.0*rs.cell().reciprocal(2) + 3.0*rs.cell().reciprocal(2) - 1.0*rs.cell().reciprocal(2));
		
		auto ff = [] (auto & kk, auto & rr){
			return exp(inq::complex(0.0, 1.0)*dot(kk, rr));
		};
		
		auto laplff = [ff] (auto & kk, auto & rr) {
			return -norm(kk)*ff(kk, rr);
		};

		CHECK(norm(kvec) == Approx(rs.cell().norm(rs.cell().to_covariant(kvec))));
		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++) func.hypercubic()[ix][iy][iz][ist] = (ist + 1.0)*ff(kvec, vec);
				}
			}
		}

		auto lapl = operations::laplacian(func, factor);
		auto func_fs = operations::transform::to_fourier(func);
		operations::laplacian_in_place(func_fs, factor);
		auto lapl_in_place = operations::transform::to_real(func_fs);
		operations::laplacian_add(operations::transform::to_fourier(func), func_fs, factor);
		auto lapl_add = operations::transform::to_real(func_fs);
		
		double diff = 0.0;
		double diff_in_place = 0.0;
		double diff_add = 0.0;		
		for(int ix = 0; ix < rs.local_sizes()[0]; ix++){
			for(int iy = 0; iy < rs.local_sizes()[1]; iy++){
				for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
					auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
					for(int ist = 0; ist < func.local_set_size(); ist++){
						auto anvalue = (ist + 1.0)*factor*laplff(kvec, vec);
						diff += fabs(lapl.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_in_place += fabs(lapl_in_place.hypercubic()[ix][iy][iz][ist] - anvalue);
						diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);						
					}
				}
			}
		}

		cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_in_place, 1, std::plus<>{});
		cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
		
		CHECK(diff < 1.0e-7) ;
		CHECK(diff_in_place < 1.0e-7);
		CHECK(diff_add < 5.0e-7);
		
	}
	
}
#endif

