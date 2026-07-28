/* -*- indent-tabs-mode: t -*- */

// Muon per-state mass fork — Tier-1 known-case test (2026-07-06).
// Oracle: for a plane wave psi = e^{ik.r}, the per-state kinetic apply
// laplacian_states gives result_ist = factor[ist]*(-|k|^2)*psi, with
// factor[ist] = -0.5*inverse_mass[ist]. Three electron states (m=1 -> -0.5)
// plus one muon state (m=206.77 -> -0.5/m). Also checks the fourier add variant
// (2x) and that the electron states are UNCHANGED by the muon slot's presence.

#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;

	auto & env = input::environment::global();

	utils::match match(1.0e-8);

	parallel::cartesian_communicator<2> cart_comm(env.comm(), {});
	auto basis_comm = basis::basis_subcomm(cart_comm);

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

	// per-state factor: muon = last GLOBAL state (nst-1), rest electrons
	auto inv_mass_of = [&](long istg){ return (istg == nst - 1) ? 1.0/m_mu : 1.0; };
	gpu::array<double, 1> factors(func.local_set_size());
	for(int ist = 0; ist < func.local_set_size(); ist++){
		auto istg = func.set_part().start() + ist;
		factors[ist] = -0.5*inv_mass_of(istg);
	}

	auto func_fs = operations::transform::to_fourier(func);
	auto lapl = operations::transform::to_real(operations::laplacian_states(func_fs, factors));

	auto acc_fs = operations::laplacian_states(operations::transform::to_fourier(func), factors);
	operations::laplacian_add_states(operations::transform::to_fourier(func), acc_fs, factors);
	auto lapl_add = operations::transform::to_real(acc_fs);

	double diff = 0.0, diff_add = 0.0, diff_elec = 0.0;
	for(int ix = 0; ix < rs.local_sizes()[0]; ix++)
		for(int iy = 0; iy < rs.local_sizes()[1]; iy++)
			for(int iz = 0; iz < rs.local_sizes()[2]; iz++){
				auto vec = rs.point_op().rvector_cartesian(ix, iy, iz);
				for(int ist = 0; ist < func.local_set_size(); ist++){
					auto istg = func.set_part().start() + ist;
					auto factor = -0.5*inv_mass_of(istg);
					auto anvalue = factor*(-norm(kvec))*ff(kvec, vec);   // factor*(-|k|^2)*psi
					diff     += fabs(lapl.hypercubic()[ix][iy][iz][ist]     - anvalue);
					diff_add += fabs(lapl_add.hypercubic()[ix][iy][iz][ist] - 2.0*anvalue);
					// electrons-unchanged: electron states must equal the pure -0.5 result
					if(istg != nst - 1){
						auto elec_value = -0.5*(-norm(kvec))*ff(kvec, vec);
						diff_elec += fabs(lapl.hypercubic()[ix][iy][iz][ist] - elec_value);
					}
				}
			}

	cart_comm.all_reduce_in_place_n(&diff, 1, std::plus<>{});
	cart_comm.all_reduce_in_place_n(&diff_add, 1, std::plus<>{});
	cart_comm.all_reduce_in_place_n(&diff_elec, 1, std::plus<>{});

	match.check("laplacian_states per-state factor",   diff,      0.0);
	match.check("laplacian_add_states (2x)",           diff_add,  0.0);
	match.check("electrons unchanged by muon slot",    diff_elec, 0.0);

	return match.fail();
}
