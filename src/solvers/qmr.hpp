/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__SOLVERS__QMR
#define INQ__SOLVERS__QMR

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

GPU_FUNCTION double safe_tol(double const & x, double const & tol) {
	return std::copysign(std::max(fabs(x), tol), x);
}

template <class operator_type, class preconditioner_type, class field_set_type>
double qmr(const operator_type & ham, const preconditioner_type & prec, field_set_type const & bb, field_set_type & xx){
	CALI_CXX_MARK_SCOPE("solver::qmr");

	double threshold = 1e-7;
	const int num_steps = 10;

	auto const nst = xx.local_set_size();
	auto const np = xx.basis().local_size();
	
	enum class qmr_status { NOT_CONVERGED, CONVERGED, BREAKDOWN_PB, BREAKDOWN_VZ, BREAKDOWN_QP, BREAKDOWN_GAMMA, RES_ZERO, B_ZERO };
	
	std::vector<qmr_status> status(nst, qmr_status::NOT_CONVERGED);
	
	auto residue = gpu::array<double, 1>(nst);

	auto exception_saved = gpu::array<complex, 2>({np, nst});
	auto saved_res = gpu::array<double, 1>(nst);
	
	auto vv = ham(xx);
	operations::shift(-1.0, bb, vv);
	auto res = vv;
	
	auto rho = operations::overlap_diagonal_sqrt(vv);
	auto norm_b = operations::overlap_diagonal_sqrt(bb);
	
	for(auto ist = 0; ist < nst; ist++) {
		if(fabs(rho[ist]) < threshold) {
			status[ist] = qmr_status::RES_ZERO;
			residue[ist] = fabs(rho[ist]);
			gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
				_exception_saved[ip][ist] = _xx[ip][ist];
			});
			saved_res[ist] = residue[ist];						
		}
	}

	for(auto ist = 0; ist < nst; ist++) {
		residue[ist] = real(rho[ist]);
		//if b is zero, the solution is trivial
		if(status[ist] == qmr_status::NOT_CONVERGED and fabs(norm_b[ist]) < 1e-16) {
			status[ist] = qmr_status::B_ZERO;
			residue[ist] = fabs(norm_b[ist]);
			gpu::run(np, [_exception_saved = begin(exception_saved), ist] GPU_LAMBDA (auto ip) {
				_exception_saved[ip][ist] = 0.0;
			});
			saved_res[ist] = residue[ist];						
		}
	}
	
	// We compute z = Pb and compute its norm \xsi^(1)
	auto zz = vv;
	prec(zz);
	auto xsi = operations::overlap_diagonal_sqrt(zz);

	auto qq = zz;
	
	auto eps = gpu::array<complex, 1>(nst, 1.0);
	auto gamma = gpu::array<double, 1>(nst, 1.0);
	auto oldgamma = gamma;
	auto eta = gpu::array<complex, 1>(nst, -1.0);
	auto alpha = gpu::array<double, 1>(nst, 1.0);
	auto theta = gpu::array<double, 1>(nst, 0.0);
	auto oldtheta = gpu::array<double, 1>(nst);
	auto beta = gpu::array<complex, 1>(nst);

	auto deltar = xx;  //just for the skeleton
	deltar.fill(0.0);
	auto deltax = xx;  //just for the skeleton 
	deltax.fill(0.0);
	
	for(int istep = 0; istep < num_steps; istep++){

		auto converged = std::all_of(status.begin(), status.end(), [] (auto sta) { return sta != qmr_status::NOT_CONVERGED; });
		if(converged) break;

		for(auto ist = 0; ist < nst; ist++) {
			if(status[ist] == qmr_status::NOT_CONVERGED and (fabs(rho[ist]) < 1e-16 or fabs(xsi[ist]) < 1e-16)) {
				gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
					_exception_saved[ip][ist] = _xx[ip][ist];
				});
				status[ist] = qmr_status::BREAKDOWN_PB;
				saved_res[ist] = residue[ist];
			}
		}

		for(auto ist = 0; ist < nst; ist++) {
			if(not (fabs(rho[ist]) < 1e-16)) {
				alpha[ist] *= xsi[ist]/rho[ist];
			} else {
				alpha [ist] = 0.0;
			}
		}

		// v^(i) = v^(i) / \rho_i
		// z^(i) = z^(i) / \xsi_i
		gpu::run(nst, np,
						 [_vv = vv.matrix().begin(), _zz = zz.matrix().begin(), _rho = rho.begin(), _xsi = xsi.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _vv[ip][ist] /= safe_tol(_rho[ist], 1e-20);
							 _zz[ip][ist] /= safe_tol(_xsi[ist], 1e-20);
						 });

		// \delta_i = v^(i)\ldotp z^(i)
		auto delta = operations::overlap_diagonal(vv, zz);

		// If \delta_i = 0, method fails
		for(auto ist = 0; ist < nst; ist++) {
			if(status[ist] == qmr_status::NOT_CONVERGED and fabs(delta[ist]) < 1e-16) {
				gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
					_exception_saved[ip][ist] = _xx[ip][ist];
				});
				status[ist] = qmr_status::BREAKDOWN_VZ;
				saved_res[ist] = residue[ist];
			}
		}

		if(istep == 0) {
			// q^(1) = z^(1)
			qq = zz;
		} else {
			// q^(i) = z^(i) - (\rho_i\delta_i)/(\eps_{i-1})q^(i-1)
			gpu::run(nst, np,
							 [_qq = qq.matrix().begin(), _zz = zz.matrix().begin(), _rho = rho.begin(), _delta = delta.begin(), _eps = eps.begin()] GPU_LAMBDA (auto ist, auto ip) {
								 if(fabs(_rho[ist]) > 1e-16) {
									 _qq[ip][ist] = _zz[ip][ist] - (_rho[ist]*_delta[ist]/_eps[ist])*_qq[ip][ist];
								 } else {
									 _qq[ip][ist] = _zz[ip][ist];
								 }
							 });
		}
		
		auto pp = ham(qq);
		// p^(i) = \alpha_{i+1} H q^(i)
		gpu::run(nst, np,
						 [_pp = pp.matrix().begin(), _alpha = alpha.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _pp[ip][ist] *= _alpha[ist];
						 });

		// \eps_i = q^{(i)}\ldotp p^{(i)}
		eps = operations::overlap_diagonal(qq, pp);

		// If \eps_i == 0, method fails
		for(auto ist = 0; ist < nst; ist++) {
			if(status[ist] == qmr_status::NOT_CONVERGED and fabs(eps[ist]) < 1e-16) {
				gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
					_exception_saved[ip][ist] = _xx[ip][ist];
				});
				status[ist] = qmr_status::BREAKDOWN_QP;
				saved_res[ist] = residue[ist];
			}
		}

		for(auto ist = 0; ist < nst; ist++) {
			if(fabs(delta[ist]) > 1e-16*fabs(eps[ist])) {
				beta[ist] = eps[ist]/delta[ist];
			} else {
				beta[ist] = 0.0;
			}
		}

		// v^(i+1) = p^(i) - \beta_i v^(i)
		gpu::run(nst, np,
						 [_pp = pp.matrix().begin(), _vv = vv.matrix().begin(), _beta = beta.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _vv[ip][ist] = _pp[ip][ist] - _beta[ist]*_vv[ip][ist];
						 });

		auto oldrho = rho;		

		// \rho_{i+1} = ||v^{i+1}||_2
		rho = operations::overlap_diagonal_sqrt(vv);

		// z^{i+1} = P v^{i+1}
		zz = vv;
		prec(zz);

		// z^{i+1} = P v^{i+1}/ \alpha^{i+1}
		gpu::run(nst, np,
						 [_zz = pp.matrix().begin(), _alpha = alpha.begin()] GPU_LAMBDA (auto ist, auto ip) {
							 _zz[ip][ist] /= (_alpha[ist] + 1e-20);
						 });

		// \xsi_{i+1} = ||z^{i+1}||_2
		xsi = operations::overlap_diagonal_sqrt(zz);

		for(auto ist = 0; ist < nst; ist++) {
			oldtheta[ist] = theta[ist];

			// \theta_i = \rho_{i+1}/(\gamma_{i-1} |\beta_i|)
			if(fabs(rho[ist]) > 1e-16) {
				theta[ist] = real(rho[ist])/(gamma[ist]*fabs(beta[ist]));
			} else {
				theta[ist] = 0.0;
			}

			oldgamma[ist] = gamma[ist];
			// \gamma_i = 1/sqrt(1+\theta_i^2)
			gamma[ist] = 1/sqrt(1 + theta[ist]*theta[ist]);

			// If \gamma_i == 0, method fails
			if(status[ist] == qmr_status::NOT_CONVERGED and fabs(gamma[ist]) < 1e-16) {
					gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
						_exception_saved[ip][ist] = _xx[ip][ist];
					});
					status[ist] = qmr_status::BREAKDOWN_GAMMA;
					saved_res[ist] = residue[ist];
			}

			// \eta_i = -\eta_{i-1}\rho_i \gamma_i^2/ (\beta_i \gamma_{i-1}^2)
			if(fabs(oldrho[ist]) > 1e-16) {
				eta[ist] = -eta[ist]*oldrho[ist]*gamma[ist]*gamma[ist]/(beta[ist]*oldgamma[ist]*oldgamma[ist]);
			} else {
				eta[ist] = 0.0;
			}
		}

		if(istep == 0) {
			gpu::run(nst, np,
							 [_deltax = deltax.matrix().begin(), _deltar = deltar.matrix().begin(), _qq = qq.matrix().begin(), _pp = pp.matrix().begin(), _eta = eta.begin(), _alpha = alpha.begin()] GPU_LAMBDA (auto ist, auto ip) {
								 // \delta_x^(1) = \eta_1 \alpha_2 q^{(1)
								 _deltax[ip][ist] = _eta[ist]*_alpha[ist]*_qq[ip][ist];
								 // \delta_r^(1) = \eta_1 p^1
								 _deltar[ip][ist] = _eta[ist]*_pp[ip][ist];
							 });
		} else {
			gpu::run(nst, np,
							 [_deltax = deltax.matrix().begin(), _deltar = deltar.matrix().begin(), _qq = qq.matrix().begin(), _pp = pp.matrix().begin(),
								_oldtheta = oldtheta.begin(), _gamma = gamma.begin(), _eta = eta.begin(), _alpha = alpha.begin()] GPU_LAMBDA (auto ist, auto ip) {

								 // \delta_x^{i} = (\theta_{i-1}\gamma_i)^2 \delta_x^{i-1} + \eta_i\alpha_{i+1} q^i
								 _deltax[ip][ist] = pow(_oldtheta[ist]*_gamma[ist], 2)*_deltax[ip][ist] + _eta[ist]*_alpha[ist]*_qq[ip][ist];
								 // \delta_r^{i} = (\theta_{i-1}\gamma_i)^2 \delta_r^{i-1} + \eta_i p^i
								 _deltar[ip][ist] = pow(_oldtheta[ist]*_gamma[ist], 2)*_deltar[ip][ist] + _eta[ist]*_pp[ip][ist];
							 });
		}

		gpu::run(nst, np,
							 [_deltax = deltax.matrix().begin(), _deltar = deltar.matrix().begin(), _xx = xx.matrix().begin(), _res = res.matrix().begin()] GPU_LAMBDA (auto ist, auto ip) {
								 // x^{i} = x^{i-1} + \delta_x^{i}
								 _xx[ip][ist]  -= _deltax[ip][ist];
								 // r^{i} = r^{i-1} - \delta_r^i
								 _res[ip][ist] -= _deltar[ip][ist];
							 });

		residue = operations::overlap_diagonal_sqrt(res);

		for(auto ist = 0; ist < nst; ist++) {
			if(fabs(norm_b[ist]) > 1e-16) residue[ist] /= norm_b[ist];
		}

		//		std::cout << istep << " --";
		//		for(auto ist = 0; ist < nst; ist++) {
		//			std::cout << '\t' << residue[ist];
		//		}
		//		std::cout << std::endl;
		
		// Convergence is reached once the residues are smaller than the threshold
		for(auto ist = 0; ist < nst; ist++) {
			if(status[ist] == qmr_status::NOT_CONVERGED and residue[ist] < threshold) status[ist] = qmr_status::CONVERGED;
		}
		
	}

	for(auto ist = 0; ist < nst; ist++) {
		if(status[ist] != qmr_status::NOT_CONVERGED and status[ist] != qmr_status::CONVERGED) {
			gpu::run(np, [_exception_saved = begin(exception_saved), _xx = begin(xx.matrix()), ist] GPU_LAMBDA (auto ip) {
				_xx[ip][ist] = _exception_saved[ip][ist];
			});
		}
		switch(status[ist]) {
		case qmr_status::BREAKDOWN_PB:
			std::cout << "State " << ist << " QMR breakdown, cannot continue: b or P*b is the zero vector!" << std::endl;
			break;
		case qmr_status::BREAKDOWN_VZ:
			std::cout << "State " << ist << " QMR breakdown, cannot continue: v^T*z is zero!" << std::endl;
			break;
		case qmr_status::BREAKDOWN_QP:
			std::cout << "State " << ist << " QMR breakdown, cannot continue: q^T*p is zero!" << std::endl;
			break;
		case qmr_status::BREAKDOWN_GAMMA:
			std::cout << "State " << ist << " QMR breakdown, cannot continue: gamma is zero!" << std::endl;
			break;
		case qmr_status::NOT_CONVERGED:
		case qmr_status::CONVERGED:
		case qmr_status::RES_ZERO:
		case qmr_status::B_ZERO:
			break;
		}

	}
	
	auto maxloc =
#ifdef HAVE_CUDA
		thrust::max_element(residue.begin(), residue.end())
#else
		std   ::max_element(residue.begin(), residue.end())
#endif
	;
	if(maxloc == residue.end()) return 0.0;
	return *maxloc;
	}

}
}
#endif

#ifdef INQ_SOLVERS_QMR_UNIT_TEST
#undef INQ_SOLVERS_QMR_UNIT_TEST

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

		const int num_iter = 10;
		
		for(int iter = 0; iter < num_iter; iter++){

			solvers::qmr(diagonal_op, identity, bb, xx);
			
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

