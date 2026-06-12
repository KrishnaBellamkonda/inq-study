/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SOLVERS__BICGSTAB
#define INQ__SOLVERS__BICGSTAB

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <math/complex.hpp>
#include <math/vector3.hpp>
#include <gpu/array.hpp>
#include <hamiltonian/ks_hamiltonian.hpp>
#include <operations/shift.hpp>
#include <operations/orthogonalize.hpp>
#include <operations/overlap_diagonal.hpp>

//required for debugging
//#include <mpi3/ostream.hpp>

namespace inq {
namespace solvers {

enum class bicgstab_status { NOT_CONVERGED, CONVERGED, BREAKDOWN_RHO_1, BREAKDOWN_W };

template <typename Comm, typename Oper, typename Preconditioner, typename Vector>
double bicgstab(Comm & comm, Oper const & oper, Preconditioner const & prec, Vector const & yy, Vector & xx) {
	CALI_CXX_MARK_SCOPE("solver::bicgstab");

	double threshold = 1e-7;
	const int num_steps = 10;

	auto const nst = xx.local_set_size();
	auto const np = xx.basis().local_size();
	
	gpu::array<bicgstab_status, 1> status(nst, bicgstab_status::NOT_CONVERGED);

	auto exception_saved = gpu::array<complex, 2>({np, nst});
	auto saved_res = gpu::array<double, 1>(nst);

	auto alpha = gpu::array<complex, 1>(nst);
	auto beta  = gpu::array<complex, 1>(nst);
	auto ww    = gpu::array<complex, 1>(nst);
	auto rho_2 = gpu::array<complex, 1>(nst);		
	
	auto hp = oper(xx);
	auto rr = hp;
	operations::shift(-1.0, yy, rr);

	auto rs = rr;

	auto gamma = operations::overlap_diagonal_sqrt(rr);

	auto pp = rr;
	
	for(int istep = 0; istep < num_steps; istep++){
		auto rho_1 = operations::overlap_diagonal(rs, rr);

		//check rho_1
		gpu::run(nst, np,
						 [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), _status = status.begin(), _rho_1 = rho_1.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 if(fabs(_rho_1[ist]) > 1e-250) return;
							 
							 _exception_saved[ip][ist] = _xx[ip][ist];

							 if(ip != 0) return;
							 
							 _status[ist] = bicgstab_status::BREAKDOWN_RHO_1;
							 _rho_1[ist] = 1.0;
						 });
		
		if(istep > 0) {
			gpu::run(nst,
							 [_beta = beta.begin(), _alpha = alpha.begin(), _rho_1 = rho_1.begin(), _rho_2 = rho_2.begin(), _ww = ww.begin()] GPU_LAMBDA (auto ist) {
								 _beta[ist] = _rho_1[ist]/_rho_2[ist]*_alpha[ist]/_ww[ist];
							 });

			gpu::run(nst, np,
							 [_pp = pp.matrix().begin(), _rr = rr.matrix().begin(), _hp = hp.matrix().begin(), _beta = beta.begin(), _ww = ww.begin()] GPU_LAMBDA (auto ist, auto ip) {
								 _pp[ip][ist] = _rr[ip][ist] + _beta[ist]*(_pp[ip][ist] - _ww[ist]*_hp[ip][ist]);
							 });
			
		}

		auto phat = pp;
		prec(phat);

		hp = oper(phat);

		alpha = operations::overlap_diagonal(rs, hp);
		gpu::run(nst,
						 [_alpha = alpha.begin(), _rho_1 = rho_1.begin()] GPU_LAMBDA (auto ist) {
							 _alpha[ist] = _rho_1[ist]/_alpha[ist];
						 });

		auto ss = rr;

		gpu::run(nst, np,
						 [_ss = ss.matrix().begin(), _hp = hp.matrix().begin(), _alpha = alpha.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _ss[ip][ist] -= _alpha[ist]*_hp[ip][ist];
						 });

		auto shat = ss;
		prec(shat);
		auto hs = oper(shat);

		ww = operations::overlap_diagonal_normalized(ss, hs, operations::conjugate{});

		gpu::run(nst, np,
						 [_xx = xx.matrix().begin(), _rr = rr.matrix().begin(), _ss = ss.matrix().begin(), _hs = hs.matrix().begin(), _shat = shat.matrix().begin(), _phat = phat.matrix().begin(),
							_alpha =  alpha.begin(), _ww = ww.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _xx[ip][ist] -= _alpha[ist]*_phat[ip][ist] + _ww[ist]*_shat[ip][ist];
							 _rr[ip][ist]  = _ss[ip][ist] - _ww[ist]*_hs[ip][ist];							 
						 });

		rho_2 = rho_1;

		gamma = operations::overlap_diagonal_sqrt(rr);
		
		//check gamma
		auto done_count = gpu::run(gpu::reduce(nst), 0,
															 [_gamma = gamma.begin(), _status = status.begin(), threshold] GPU_LAMBDA (auto ist) {
																 if(fabs(_gamma[ist]) < threshold) _status[ist] = bicgstab_status::CONVERGED;
																 if(_status[ist] != bicgstab_status::NOT_CONVERGED) return 1;
																 return 0;
															 });

		auto done = comm.all_reduce_value(done_count == nst);
		if(done) break;

		//check w
		gpu::run(nst, np,
						 [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), _status = status.begin(), _ww = ww.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 if(fabs(_ww[ist]) > 1e-250) return;
							 
							 _exception_saved[ip][ist] = _xx[ip][ist];
							 
							 if(ip != 0) return;
							 
							 _status[ist] = bicgstab_status::BREAKDOWN_W;
							 _ww[ist] = 1.0;
						 });
	}
	
	for(auto ist = 0; ist < nst; ist++) {
		if(status[ist] != bicgstab_status::NOT_CONVERGED and status[ist] != bicgstab_status::CONVERGED) {
			gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
				_xx[ip][ist] = _exception_saved[ip][ist];
			});
		}
		switch(status[ist]) {
		case bicgstab_status::BREAKDOWN_RHO_1:
			std::cout << "State " << ist << " BICGSTAB breakdown, cannot continue: rho_1 is zero!" << std::endl;
			break;
		case bicgstab_status::BREAKDOWN_W:
			std::cout << "State " << ist << " BICGSTAB breakdown, cannot continue: w is zero!" << std::endl;
			break;
		case bicgstab_status::NOT_CONVERGED:
		case bicgstab_status::CONVERGED:
			break;
		}

	}
 
	auto maxloc =
#ifdef HAVE_GPU
		thrust::
#else
		std::
#endif
		max_element(gamma.begin(), gamma.end());
		
	auto max_val = 0.0;
	if(maxloc != gamma.end()) max_val = *maxloc;
	
	return comm.max(max_val);
}

template <typename Oper, typename Preconditioner, typename Vector>
auto bicgstab(Oper const & oper, Preconditioner const & prec, Vector const & yy, Vector & xx) {
	auto comm = boost::mpi3::environment::get_self_instance();
	return bicgstab(comm, oper, prec, yy, xx);
}

}
}
#endif

#ifdef INQ_SOLVERS_BICGSTAB_UNIT_TEST
#undef INQ_SOLVERS_BICGSTAB_UNIT_TEST

#include <basis/trivial.hpp>
#include <operations/matrix_operator.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	
  const int npoint = 100;
  const int nvec = 12;

  basis::trivial bas(npoint, boost::mpi3::environment::get_self_instance());
	
	gpu::array<complex, 2> identity_matrix({npoint, npoint});
  
	for(int ip = 0; ip < npoint; ip++){
		for(int jp = 0; jp < npoint; jp++){
			identity_matrix[ip][jp] = 0.0;
			if(ip == jp) identity_matrix[ip][jp] = 1.0;
		}
	}
  
	operations::matrix_operator<complex> identity(std::move(identity_matrix));

	SECTION("Diagonal matrix complex"){
  
    gpu::array<complex, 2> diagonal_matrix({npoint, npoint});
    
    for(int ip = 0; ip < npoint; ip++){
      for(int jp = 0; jp < npoint; jp++){
        diagonal_matrix[ip][jp] = 0.0;
        if(ip == jp) diagonal_matrix[ip][jp] = ip + 1.0;
      }
    }
    
    operations::matrix_operator<complex> diagonal_op(std::move(diagonal_matrix));
    
    basis::field_set<basis::trivial, complex> xx(bas, nvec);
    basis::field_set<basis::trivial, complex> bb(bas, nvec);
 
		for(int ip = 0; ip < npoint; ip++){
      for(int ivec = 0; ivec < nvec; ivec++){
        xx.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
        bb.matrix()[ip][ivec] = cos(ip*(ivec + 1.0)/2.0);
      }
    }

		const int num_iter = 20;
		
		for(int iter = 0; iter < num_iter; iter++){

			solvers::bicgstab(diagonal_op, identity, bb, xx);
			
			auto residual = diagonal_op(xx);
			operations::shift(-1.0, bb, residual);
			auto normres = operations::overlap_diagonal(residual);
      
      fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
			
      for(int ivec = 0; ivec < xx.set_size(); ivec++){
				fmt::print(std::cout, "    state {:4d}  res = {:15.10e}\n", ivec + 1, real(normres[ivec]));
			}

			if(num_iter - 1 == iter) {
				for(int ivec = 0; ivec < xx.set_size(); ivec++)	{
					CHECK(fabs(normres[ivec]) < 1e-8);
				}
			}
		}
 }


}
#endif

