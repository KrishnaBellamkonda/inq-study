/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <iostream>

#include <sharmonic.h>

#include <complex>
#include <cassert>

#include <catch2/catch_all.hpp>
#include <fstream>
#include <vector>

#ifdef ENABLE_GPU
#include <thrust/complex.h>
using complex = thrust::complex<double>;
#else
#include <complex>
using complex = std::complex<double>;
#endif

SH_GPU_FUNCTION double real(const complex & z){
	return z.real();
}

SH_GPU_FUNCTION auto imag(const complex & z){
	return z.imag();
}

template <typename Function>
auto finite_difference_gradient(Function const & func) {
	auto const spacing = 1e-6;
			
	auto grad = 0.0;
	grad += -1/12.0*func( 2*spacing);
	grad +=  8/12.0*func(   spacing);
	grad += -8/12.0*func(  -spacing);
	grad +=  1/12.0*func(-2*spacing);

	return grad/spacing;
}

template <typename Grad>
auto check(int ll, int mm, Grad const & num_grad, Grad const & grad, double const & tol) {

	auto tot_diff = 0.0;
	
	for(auto idir = 0; idir < 3; idir++) {
		auto diff = std::abs(num_grad[idir] - grad[idir]);
		
		if(diff > tol) {
			std::cout << ll << '\t' << mm << '\t' << idir << '\t' << grad[idir] << '\t' << num_grad[idir] << '\t' << diff << std::endl;
		}
		
		tot_diff += diff;
	}

	return tot_diff/3.0;

}

auto compare() {

	auto const tol = 2e-8;
	auto reps = 10000;
	
	for(int ll = 1; ll <= 9; ll++) {
		for(int mm = -ll; mm <= ll; mm++) {
			
			srand48(0xFC769E60887BB49Cl);

			auto diff_cart = 0.0;
			auto diff_norm = 0.0;
			auto diff_ang  = 0.0;
			auto diff_hess = 0.0;
			auto count = 0;
				
			for(int iter = 0; iter < reps; iter++){
				double xx = 2.0*(drand48() - 0.5);
				double yy = 2.0*(drand48() - 0.5);
				double zz = 2.0*(drand48() - 0.5);

				if(xx*xx + yy*yy + zz*zz > 1.0) continue;

				count++;
				
				double rr, theta, phi;
				sharmonic::internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
				
				std::array<double, 3> num_grad;
				num_grad[0] = rr*finite_difference_gradient([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx + pos, yy, zz); });
				num_grad[1] = rr*finite_difference_gradient([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx, yy + pos, zz); });
				num_grad[2] = rr*finite_difference_gradient([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx, yy, zz + pos); });

				std::array<double, 3> cart_grad;
				sharmonic::cartesian_real_rgradient(ll, mm, xx, yy, zz, &cart_grad[0], &cart_grad[1], &cart_grad[2]);

				diff_cart += check(ll, mm, num_grad, cart_grad, tol);

				assert(not std::isnan(diff_cart));
	
				sharmonic::internal::normalize(&xx, &yy, &zz);

				std::array<double, 3> norm_grad;
				sharmonic::normalized_cartesian_real_rgradient(ll, mm, xx, yy, zz, &norm_grad[0], &norm_grad[1], &norm_grad[2]);
				diff_norm += check(ll, mm, num_grad, norm_grad, tol);

				assert(not std::isnan(diff_norm));
				
				std::array<double, 3> ang_grad;
				sharmonic::angular_real_rgradient(ll, mm, theta, phi, &ang_grad[0], &ang_grad[1], &ang_grad[2]);
				diff_ang += check(ll, mm, num_grad, ang_grad, tol);

				double hxx, hyy, hzz, hxy, hxz, hyz;
				std::array<double, 3> hess_grad;
				sharmonic::normalized_cartesian_real_r2hessian(ll, mm, xx, yy, zz, &hess_grad[0], &hess_grad[1], &hess_grad[2], &hxx, &hyy, &hzz, &hxy, &hxz, &hyz);
				diff_hess += check(ll, mm, num_grad, hess_grad, tol);
			}

			CHECK(diff_cart < count*tol);
			CHECK(diff_norm < count*tol);
			CHECK(diff_ang  < count*tol);
			CHECK(diff_hess < count*tol);

		}
	}
}

TEST_CASE("reference", "[reference]") {
	using namespace Catch::literals;
	using Catch::Approx;

	using namespace sharmonic;
	using sharmonic::internal::normalize;
	using sharmonic::cartesian_real;
	using sharmonic::cartesian_complex;

	SECTION("reference values") {
		compare();
	}

}
