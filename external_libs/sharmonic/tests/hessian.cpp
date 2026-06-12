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
auto finite_difference_second_order(Function const & func) {
	auto const spacing = 1e-4;
	
	auto d2 = 0.0;
	d2 +=  -1*func(-2*spacing);	
	d2 +=  16*func(-1*spacing);
	d2 += -30*func(         0);
	d2 +=  16*func( 1*spacing);	
	d2 +=  -1*func( 2*spacing);
	return d2/(12*spacing*spacing);
}

template <typename Function>
auto finite_difference_second_order_cross(Function const & func) {
	auto const hh = 1e-4;
	
	auto d2 = 0.0;
	d2 += -1*func(-2*hh,  2*hh) +  8*func(-1*hh,  2*hh) -  8*func( 1*hh,  2*hh) + 1*func( 2*hh,  2*hh);
	d2 +=  8*func(-2*hh,  1*hh) - 64*func(-1*hh,  1*hh) + 64*func( 1*hh,  1*hh) - 8*func( 2*hh,  1*hh);
	d2 += -8*func(-2*hh, -1*hh) + 64*func(-1*hh, -1*hh) - 64*func( 1*hh, -1*hh) + 8*func( 2*hh, -1*hh);
	d2 +=  1*func(-2*hh, -2*hh) -  8*func(-1*hh, -2*hh) +  8*func( 1*hh, -2*hh) - 1*func( 2*hh, -2*hh);
	return d2/(144*hh*hh);
}

template <typename Hessian>
auto check(int ll, int mm, Hessian const & num_hessian, Hessian const & hessian, double const & tol) {

	std::vector<std::string> labels({"xx", "yy", "zz", "xy", "xz", "yz"});
	
	auto tot_diff = 0.0;
	
	for(auto idir = 0; idir < 6; idir++) {
		auto diff = std::abs(num_hessian[idir] - hessian[idir]);
		
		if(diff > tol) {
			std::cout << ll << '\t' << mm << '\t' << labels[idir] << '\t' << hessian[idir] << '\t' << num_hessian[idir] << '\t' << diff << std::endl;
		}
		
		tot_diff += diff;
	}
	
	return tot_diff/6.0;

}

auto compare() {

	auto const tol = 1e-7;
	auto reps = 2;
	
	for(int ll = 0; ll <= 9; ll++) {
		for(int mm = -ll; mm <= ll; mm++) {
			std::cout << ll << '\t' << mm << std::endl;
			srand48(0xFC769E60887BB49Cl);

			auto diff_ang  = 0.0;
			auto diff_norm = 0.0;
			auto count = 0;
				
			for(int iter = 0; iter < reps; iter++){
				double xx = 2.0*(drand48() - 0.5);
				double yy = 2.0*(drand48() - 0.5);
				double zz = 2.0*(drand48() - 0.5);

				if(xx*xx + yy*yy + zz*zz > 1.0) continue;

				count++;
				
				double rr, theta, phi;
				sharmonic::internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
				
				std::array<double, 6> num_hessian;
				num_hessian[0] = rr*rr*finite_difference_second_order([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx + pos, yy, zz); });
				num_hessian[1] = rr*rr*finite_difference_second_order([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx, yy + pos, zz); });
				num_hessian[2] = rr*rr*finite_difference_second_order([&](auto pos) { return sharmonic::cartesian_real(ll, mm, xx, yy, zz + pos); });
				num_hessian[3] = rr*rr*finite_difference_second_order_cross([&](auto pos1, auto pos2) { return sharmonic::cartesian_real(ll, mm, xx + pos1, yy + pos2, zz); });
				num_hessian[4] = rr*rr*finite_difference_second_order_cross([&](auto pos1, auto pos2) { return sharmonic::cartesian_real(ll, mm, xx + pos1, yy, zz + pos2); });
				num_hessian[5] = rr*rr*finite_difference_second_order_cross([&](auto pos1, auto pos2) { return sharmonic::cartesian_real(ll, mm, xx, yy + pos1, zz + pos2); });
				/*
				std::cout << "COORDS " << xx << '\t' << yy << '\t' << zz << std::endl;
				std::cout << "xx\t" << num_hessian[0]/rr/rr << std::endl;
				std::cout << "yy\t" << num_hessian[1]/rr/rr << std::endl;
				std::cout << "zz\t" << num_hessian[2]/rr/rr << std::endl;
				std::cout << "xy\t" << num_hessian[3]/rr/rr << std::endl;
				std::cout << "xz\t" << num_hessian[4]/rr/rr << std::endl;
				std::cout << "yz\t" << num_hessian[5]/rr/rr << std::endl;
				*/
				sharmonic::internal::normalize(&xx, &yy, &zz);
				
				std::array<double, 6> norm_hessian;
				double gx, gy, gz;
				sharmonic::normalized_cartesian_real_r2hessian(ll, mm, xx, yy, zz, &gx, &gy, &gz, &norm_hessian[0], &norm_hessian[1], &norm_hessian[2], &norm_hessian[3], &norm_hessian[4], &norm_hessian[5]);
				diff_norm += check(ll, mm, num_hessian, norm_hessian, tol);
				
			}

			CHECK(diff_ang  < count*tol);
			CHECK(diff_norm < count*tol);

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
