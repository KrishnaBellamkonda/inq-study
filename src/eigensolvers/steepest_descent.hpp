/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__EIGENSOLVERS__STEEPEST_DESCENT
#define INQ__EIGENSOLVERS__STEEPEST_DESCENT

// Copyright (C) 2019-2026 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
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

namespace inq {
namespace eigensolvers {

template <typename Ham, typename Metric, typename Prec, typename field_set_type>
void steepest_descent(Ham const & ham, Metric const & metric, Prec const & prec, field_set_type & phi){

	CALI_CXX_MARK_FUNCTION;
	
	const int num_steps = 10;

	for(int istep = 0; istep < num_steps; istep++){

		auto hphi = ham(phi); //this shouldn't be recalculated, it can be obtained when getting the new phi (otherwise the algorithm is twice as slow)
		auto residual = hphi;
		auto eigenvalues = operations::overlap_diagonal(residual, phi);
		auto mphi = metric(phi);
		auto norm =	operations::overlap_diagonal(mphi,phi);
			
		auto evnorm = eigenvalues;
		
		gpu::run(eigenvalues.size(),
						 [evn = begin(evnorm), nor = begin(norm)] GPU_LAMBDA (auto ist){
							 evn[ist] /= real(nor[ist]);
						 });
		
		operations::shift(-1.0, evnorm, mphi, residual);

		prec(residual);

		auto mm = gpu::array<typename field_set_type::element_type, 2>({6, eigenvalues.size()});

		auto hresidual = ham(residual);
		auto mresidual = metric(residual);
		mm[0] = operations::overlap_diagonal(mresidual, residual);
		mm[1] = operations::overlap_diagonal(mphi, residual);
		mm[2] = operations::overlap_diagonal(residual, hresidual);
		mm[3] = operations::overlap_diagonal(phi, hresidual);
		mm[4] = eigenvalues;
		mm[5] = norm;

		auto lambda = gpu::array<double, 1>(eigenvalues.size());
		
		gpu::run(phi.local_spinor_set_size(),
						 [m = begin(mm), lam = begin(lambda)]
						 GPU_LAMBDA (auto ist){
							 auto ca = real(m[0][ist]*m[3][ist] - m[2][ist]*m[1][ist]);
							 auto cb = real(m[5][ist]*m[2][ist] - m[4][ist]*m[0][ist]);
							 auto cc = real(m[4][ist]*m[1][ist] - m[3][ist]*m[5][ist]);
							 auto den = cb + sqrt(cb*cb - 4.0*ca*cc);

							 if(fabs(den) < 1e-15) { //this happens if we are perfectly converged
								 lam[ist] = 0.0;
							 } else {
								lam[ist] = 2.0*cc/den;
							 }
						 });

		gpu::run(phi.local_spinor_set_size(), phi.spinor_matrix().size(),
						 [res = begin(residual.spinor_matrix()), ph = begin(phi.spinor_matrix()),
							hres = begin(hresidual.spinor_matrix()), hph = begin(hphi.spinor_matrix()), lam = begin(lambda)] GPU_LAMBDA (auto ist, auto ip){
							 ph[ip][ist] += lam[ist]*res[ip][ist];
						 });

		operations::orthogonalize(phi, metric);
	}

	// operations::orthogonalize(phi, metric);
		
}

template <class operator_type, class preconditioner_type, class field_set_type>
void steepest_descent(const operator_type & ham, const preconditioner_type & prec, field_set_type & phi){

	CALI_CXX_MARK_FUNCTION;
	
	const int num_steps = 5;

	auto hphi = ham(phi);
	
	for(int istep = 0; istep < num_steps; istep++){

		auto residual = hphi;
		auto eigenvalues = operations::overlap_diagonal(residual, phi);
		auto norm =	operations::overlap_diagonal(phi);
			
		auto evnorm = eigenvalues;
		
		gpu::run(eigenvalues.size(),
						 [evn = begin(evnorm), nor = begin(norm)] GPU_LAMBDA (auto ist){
							 evn[ist] /= real(nor[ist]);
						 });
		
		operations::shift(-1.0, evnorm, phi, residual);

		prec(residual);

		auto mm = gpu::array<typename field_set_type::element_type, 2>({6, eigenvalues.size()});

		auto hresidual = ham(residual);
		mm[0] = operations::overlap_diagonal(residual, residual);
		mm[1] = operations::overlap_diagonal(phi, residual);
		mm[2] = operations::overlap_diagonal(residual, hresidual);
		mm[3] = operations::overlap_diagonal(phi, hresidual);
		mm[4] = eigenvalues;
		mm[5] = norm;

		auto lambda = gpu::array<double, 1>(eigenvalues.size());
		
		gpu::run(phi.local_spinor_set_size(),
						 [m = begin(mm), lam = begin(lambda)]
						 GPU_LAMBDA (auto ist){
							 auto ca = real(m[0][ist]*m[3][ist] - m[2][ist]*m[1][ist]);
							 auto cb = real(m[5][ist]*m[2][ist] - m[4][ist]*m[0][ist]);
							 auto cc = real(m[4][ist]*m[1][ist] - m[3][ist]*m[5][ist]);
							 auto den = cb + sqrt(cb*cb - 4.0*ca*cc);

							 if(fabs(den) < 1e-15) { //this happens if we are perfectly converged
								 lam[ist] = 0.0;
							 } else {
								 lam[ist] = 2.0*cc/den;
							 }
						 });

		gpu::run(phi.local_spinor_set_size(), phi.spinor_matrix().size(),
						 [res = begin(residual.spinor_matrix()), ph = begin(phi.spinor_matrix()),
							hres = begin(hresidual.spinor_matrix()), hph = begin(hphi.spinor_matrix()), lam = begin(lambda), istep] GPU_LAMBDA (auto ist, auto ip){
							 ph[ip][ist] += lam[ist]*res[ip][ist];
							 if(istep !=  num_steps - 1) hph[ip][ist] += lam[ist]*hres[ip][ist];
						 });
	}

	operations::orthogonalize(phi);
		
}

}
}
#endif

#ifdef INQ_EIGENSOLVERS_STEEPEST_DESCENT_UNIT_TEST
#undef INQ_EIGENSOLVERS_STEEPEST_DESCENT_UNIT_TEST

#include <basis/trivial.hpp>
#include <operations/matrix_operator.hpp>

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

	using namespace inq;
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("Diagonal matrix complex eigenvalue problem"){

		const int npoint = 100;
		const int nvec = 12;
		
		basis::trivial bas(npoint, parallel::communicator{boost::mpi3::environment::get_self_instance()});
		
		gpu::array<complex, 2> identity_matrix({npoint, npoint});
	
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				identity_matrix[ip][jp] = 0.0;
				if(ip == jp) identity_matrix[ip][jp] = 1.0;
			}
		}
	
		operations::matrix_operator<complex> identity(std::move(identity_matrix));

	
		gpu::array<complex, 2> diagonal_matrix({npoint, npoint});
		
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				diagonal_matrix[ip][jp] = 0.0;
				if(ip == jp) diagonal_matrix[ip][jp] = ip + 1.0;
			}
		}
		
		operations::matrix_operator<complex> diagonal_op(std::move(diagonal_matrix));
		
		basis::field_set<basis::trivial, complex> phi(bas, nvec);

		for(int ip = 0; ip < npoint; ip++){
			for(int ivec = 0; ivec < nvec; ivec++){
				phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
			}
		}

		operations::orthogonalize(phi);

		const int num_iter = 100;
		
		for(int iter = 0; iter < num_iter; iter++){

			eigensolvers::steepest_descent(diagonal_op, identity, phi);
			
			auto residual = diagonal_op(phi);
			auto eigenvalues = operations::overlap_diagonal(phi, residual);
			operations::shift(-1.0, eigenvalues, phi, residual);
			auto normres = operations::overlap_diagonal(residual);
			
			/*
				fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
				
				for(int ivec = 0; ivec < phi.local_set_size(); ivec++){
				fmt::print(std::cout, "    state {:4d}  evalue = {:18.12f}  res = {:15.10e}\n", ivec + 1, real(eigenvalues[ivec]), real(normres[ivec]));
				}
			*/
		
			if(num_iter - 1 == iter){

				CHECK(fabs(eigenvalues[0]) == 1.000000001634_a);
				CHECK(fabs(eigenvalues[1]) == 2.000000003689_a);
				CHECK(fabs(eigenvalues[2]) == 3.000000001955_a);
				CHECK(fabs(eigenvalues[3]) == 4.000000001760_a);
				CHECK(fabs(eigenvalues[4]) == 5.000000002561_a);
				CHECK(fabs(eigenvalues[5]) == 6.000000003127_a);
				CHECK(fabs(eigenvalues[6]) == 7.000000002312_a);
				CHECK(fabs(eigenvalues[7]) == 8.000000000292_a);
				CHECK(fabs(eigenvalues[8]) == 8.999999999033_a);
				CHECK(fabs(eigenvalues[9]) == 9.999999998497_a);
				CHECK(fabs(eigenvalues[10]) == 10.999999998768_a);
				CHECK(fabs(eigenvalues[11]) == 11.999999998422_a);

			}
		}
	}
	
	SECTION("Tridiagonal matrix complex eigenvalue problem"){

		const int npoint = 10;
		const int nvec = 5;
		
		basis::trivial bas(npoint, parallel::communicator{boost::mpi3::environment::get_self_instance()});
		
		gpu::array<complex, 2> identity_matrix({npoint, npoint});
	
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				identity_matrix[ip][jp] = 0.0;
				if(ip == jp) identity_matrix[ip][jp] = 1.0;
			}
		}
	
		operations::matrix_operator<complex> identity(std::move(identity_matrix));

	
		gpu::array<complex, 2> tridiagonal_matrix({npoint, npoint});
		
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				tridiagonal_matrix[ip][jp] = 0.0;
				if(ip == jp) tridiagonal_matrix[ip][jp] = ip + 1.0;
				if(ip == jp - 1 || ip == jp + 1) tridiagonal_matrix[ip][jp] = -2.0;
			}
		}
		
		operations::matrix_operator<complex> tridiagonal_op(std::move(tridiagonal_matrix));
		
		basis::field_set<basis::trivial, complex> phi(bas, nvec);

		for(int ip = 0; ip < npoint; ip++){
			for(int ivec = 0; ivec < nvec; ivec++){
				phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.1));
			}
		}

		operations::orthogonalize(phi);

		const int num_iter = 100;
		
		for(int iter = 0; iter < num_iter; iter++){

			eigensolvers::steepest_descent(tridiagonal_op, identity, phi);
			
			auto residual = tridiagonal_op(phi);
			auto eigenvalues = operations::overlap_diagonal(phi, residual);
			operations::shift(-1.0, eigenvalues, phi, residual);
			auto normres = operations::overlap_diagonal(residual);

			/*
				fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
				
				for(int ivec = 0; ivec < phi.local_set_size(); ivec++){
				fmt::print(std::cout, "    state {:4d}  evalue = {:18.12f}  res = {:15.10e}\n", ivec + 1, real(eigenvalues[ivec]), real(normres[ivec]));
				}
			*/

			if(num_iter - 1 == iter){

				CHECK(Approx(real(eigenvalues[0])).epsilon(1e-9) == -1.125485797042);
				CHECK(Approx(real(eigenvalues[1])).epsilon(1e-9) ==  0.908833398967);
				CHECK(Approx(real(eigenvalues[2])).epsilon(1e-9) ==  2.485046549295);
				CHECK(Approx(real(eigenvalues[3])).epsilon(1e-9) ==  3.801223280185);
				CHECK(Approx(real(eigenvalues[4])).epsilon(1e-9) ==  4.953794415028);

			}
		}
	}

	SECTION("Generalized complex eigenvalue problem with Tridigonal matrix and diagonal metric"){

		const int npoint = 5;
		const int nvec = 5;
		
		basis::trivial bas(npoint, parallel::communicator{boost::mpi3::environment::get_self_instance()});
		
		gpu::array<complex, 2> identity_matrix({npoint, npoint});
	
		for(int ip = 0; ip < npoint; ip++){
			for(int jp = 0; jp < npoint; jp++){
				identity_matrix[ip][jp] = 0.0;
				if(ip == jp) identity_matrix[ip][jp] = 1.0;
			}
		}
	
		operations::matrix_operator<complex> identity(std::move(identity_matrix));

	
		gpu::array<complex, 2> matrix({npoint, npoint});
		matrix[0][0] = 0.020010738738616; matrix[0][1] = 0.774873729966280; matrix[0][2] = 0.000000000000000; matrix[0][3] = 0.000000000000000; matrix[0][4] = 0.000000000000000;
		matrix[1][0] = 0.774873729966280; matrix[1][1] = 0.689666215444812; matrix[1][2] = 0.486278931258996; matrix[1][3] = 0.000000000000000; matrix[1][4] = 0.000000000000000;
		matrix[2][0] = 0.000000000000000; matrix[2][1] = 0.486278931258996; matrix[2][2] = 0.823127835294756; matrix[2][3] = 0.300637349284769; matrix[2][4] = 0.000000000000000;
		matrix[3][0] = 0.000000000000000; matrix[3][1] = 0.000000000000000; matrix[3][2] = 0.300637349284769; matrix[3][3] = 0.467146725505689; matrix[3][4] = 0.888330694944989;
		matrix[4][0] = 0.000000000000000; matrix[4][1] = 0.000000000000000; matrix[4][2] = 0.000000000000000; matrix[4][3] = 0.888330694944989; matrix[4][4] = 0.117382906014399;
		operations::matrix_operator<complex> matrix_op(std::move(matrix));

		gpu::array<complex, 2> metric({npoint, npoint});
		metric[0][0] = 1.000000000000000; metric[0][1] = 0.000000000000000; metric[0][2] = 0.000000000000000; metric[0][3] = 0.000000000000000; metric[0][4] = 0.000000000000000;
		metric[1][0] = 0.000000000000000; metric[1][1] = 0.644865661048916; metric[1][2] = 0.000000000000000; metric[1][3] = 0.000000000000000; metric[1][4] = 0.000000000000000;
		metric[2][0] = 0.000000000000000; metric[2][1] = 0.000000000000000; metric[2][2] = 0.848303045935845; metric[2][3] = 0.000000000000000; metric[2][4] = 0.000000000000000;
		metric[3][0] = 0.000000000000000; metric[3][1] = 0.000000000000000; metric[3][2] = 0.000000000000000; metric[3][3] = 0.683322128168945; metric[3][4] = 0.000000000000000;
		metric[4][0] = 0.000000000000000; metric[4][1] = 0.000000000000000; metric[4][2] = 0.000000000000000; metric[4][3] = 0.000000000000000; metric[4][4] = 0.573762177561172;
		operations::matrix_operator<complex> metric_op(std::move(metric));
		
		basis::field_set<basis::trivial, complex> phi(bas, nvec);

		for(int ip = 0; ip < npoint; ip++){
			for(int ivec = 0; ivec < nvec; ivec++){
				phi.matrix()[ip][ivec] = exp(complex(0.0, (ip*ivec)*0.01));
			}
		}

		operations::orthogonalize(phi,metric_op);

		const int num_iter = 100;
		
		for(int iter = 0; iter < num_iter; iter++){

			eigensolvers::steepest_descent(matrix_op, metric_op, identity, phi);
			
			auto residual = matrix_op(phi);
			auto eigenvalues = operations::overlap_diagonal(phi, residual);
			auto mphi = metric_op(phi);
			operations::shift(-1.0, eigenvalues, mphi, residual);
			auto normres = operations::overlap_diagonal(residual);

			/*
				fmt::print(std::cout, "  Iteration {:4d}:\n", iter);
				
				for(int ivec = 0; ivec < phi.local_set_size(); ivec++){
				fmt::print(std::cout, "    state {:4d}  evalue = {:18.12f}  res = {:15.10e}\n", ivec + 1, real(eigenvalues[ivec]), real(normres[ivec]));
				}
			*/

			if(num_iter - 1 == iter){
				CHECK(Approx(real(eigenvalues[0])).epsilon(1e-10) == -1.034793846252087);
				CHECK(Approx(real(eigenvalues[1])).epsilon(1e-10) == -0.624485836749884);
				CHECK(Approx(real(eigenvalues[2])).epsilon(1e-10) == 0.688895402957676);
				CHECK(Approx(real(eigenvalues[3])).epsilon(1e-10) == 1.806483545521078);
				CHECK(Approx(real(eigenvalues[4])).epsilon(1e-10) == 2.111932189253582);
			}
		}
	}
}
#endif
