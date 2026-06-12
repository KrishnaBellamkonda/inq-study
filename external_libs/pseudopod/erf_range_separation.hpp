/* -*- indent-tabs-mode: t -*- */

#ifndef MATH__ERF_RANGE_SEPARATION
#define MATH__ERF_RANGE_SEPARATION

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cmath>

#include <gpu/run.hpp>

namespace pseudo {
namespace math {
class erf_range_separation {
public:
    
	erf_range_separation(double sigma):
		sigma_(sigma){
	}

	auto long_range_potential(double rr) const {
		if(rr < 1e-8) return -2.0/(sqrt(2.0*M_PI)*sigma_);
		return -erf(rr/(sigma_*sqrt(2.0)))/rr;
	}

	auto short_range_potential(double rr) const {
		return -erfc(rr/(sigma_*sqrt(2.0)))/rr;
	}

	auto short_range_potential_radius() const {

		/*
			This code is numerically unstable for small values of tol, so I am replacing it by a hard-wired value 
			
			// find the inverse of the function by Newton-Raphson
			double x = 1.0;
			double xold = 0.0;
			for(int ii = 0; ii < 100; ii++){
			double f = erf(x/(sigma_*sqrt(2.0))) + tol*x - 1.0;
			double df = sqrt(2.0/M_PI)*exp(-0.5*square(x/sigma_))/sigma_ + tol;
			xold = x;
			x = x - f/df;
			if(fabs(xold - x) < 1e-5) break;
			}

			return x;

		*/

		return 1.593078562;
	}

	GPU_FUNCTION auto long_range_density(double rr) const {
		const double exp_arg = -0.5*square(rr/sigma_);
		return -exp(exp_arg)/cube(sigma_*sqrt(2.0*M_PI));
	}

	auto long_range_density_radius(double tol = 1e-15) const {
		return sqrt(-2.0*log(cube(sigma_*sqrt(2.0*M_PI))*tol))*sigma_;
	}
		
	auto long_range_density_fourier(double gg) const {
		return -exp(-0.5*square(gg*sigma_));
	}

	auto long_range_density_fourier_radius(double tol = 1e-15) const {
		return sqrt(-2.0*log(tol))/sigma_;
	}

	auto self_interaction() const{
		return 1.0/(2.0*sqrt(M_PI)*sigma_);
	}

	auto sigma() const {
		return sigma_;
	}

	auto alpha() const {
		return 1.0/(sqrt(2.0)*sigma_);
	}

private:
		
	GPU_FUNCTION static double square(double x){
		return x*x;
	}
		
	GPU_FUNCTION static double cube(double x){
		return x*x*x;
	}
    
	double sigma_;
    
};

}
}

#ifdef INQ__ERF_RANGE_SEPARATION_UNIT_TEST
#undef INQ__ERF_RANGE_SEPARATION_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
	
  auto sep = pseudo::math::erf_range_separation(0.2);

	SECTION("Potential"){

		CHECK(sep.short_range_potential(0.4) + sep.long_range_potential(0.4) == Approx(-1.0/0.4));
		CHECK(sep.short_range_potential(3.3) + sep.long_range_potential(3.3) == Approx(-1.0/3.3));

		CHECK(sep.long_range_potential(0.0) == Approx(-3.989422804));
		CHECK(sep.long_range_potential(1.0e-7) == Approx(-3.989422804));
		CHECK(sep.long_range_potential(1.0e-3) == Approx(-3.9894061815));
		CHECK(sep.long_range_potential(1.0e-1) == Approx(-3.8292492255));
		CHECK(sep.long_range_potential(5.0e-1) == Approx(-1.9751613387));
		CHECK(sep.long_range_potential(1.0) == Approx(-0.9999994267));
		CHECK(sep.long_range_potential(2.0) == Approx(-0.5));
		CHECK(sep.long_range_potential(3.0) == Approx(-0.3333333333));
		CHECK(sep.long_range_potential(4.0) == Approx(-0.25));
		CHECK(sep.long_range_potential(5.0) == Approx(-0.2));
		CHECK(sep.long_range_potential(6.0) == Approx(-0.16666667));
		CHECK(sep.long_range_potential(7.0) == Approx(-0.14285714));
		CHECK(sep.long_range_potential(8.0) == Approx(-0.125));
		CHECK(sep.long_range_potential(9.0) == Approx(-0.1111111111));
		CHECK(sep.long_range_potential(10.0) == Approx(-0.1));

		CHECK(sep.short_range_potential(1.0e-7) == Approx(-9999996.0105771963));
		CHECK(sep.short_range_potential(1.0e-3) == Approx(-996.0105938185));
		CHECK(sep.short_range_potential(1.0e-1) == Approx(-6.1707507745));
		CHECK(sep.short_range_potential(2.5e-1) == Approx(-0.8451981893));
		CHECK(sep.short_range_potential(5.0e-1) == Approx(-0.0248386613));
		CHECK(sep.short_range_potential(7.5e-1) == Approx(-0.0002357794));
		CHECK(sep.short_range_potential(1.0) == Approx(-0.0000005733));
		CHECK(sep.short_range_potential(1.25) == Approx(-3.283621e-10));
		CHECK(sep.short_range_potential(1.5) == Approx(-4.254522e-14));
		CHECK(fabs(sep.short_range_potential(1.75)) <= 1e-15);
		CHECK(fabs(sep.short_range_potential(2.0)) <= 1e-15);
		CHECK(fabs(sep.short_range_potential(2.5)) <= 1e-15);
		CHECK(fabs(sep.short_range_potential(3.0)) <= 1e-15);
		CHECK(fabs(sep.short_range_potential(3.5)) <= 1e-15);
		CHECK(fabs(sep.short_range_potential(4.0)) <= 1e-15);
	}

	SECTION("Short range potential radius"){

		auto radius = sep.short_range_potential_radius();
		CHECK(radius == 1.593078562_a);
		CHECK(sep.short_range_potential(radius) == -1.033846e-15_a);
	
	}

	SECTION("Real space density"){

		CHECK(sep.long_range_density_radius() == 1.7113801339_a);
		
		// the integral must be -1.0
		double dx = 0.1;
		double integral = 0.0;
		for(double x = dx; x < 100.0; x += dx) integral += dx*4.0*M_PI*x*x*sep.long_range_density(x);

		CHECK(sep.long_range_density(0.0) == -7.9367044918_a);
		CHECK(integral == -1.0_a);

	}

	SECTION("Fourier space density"){

		CHECK(sep.long_range_density_fourier_radius() == 41.5564534067_a);
		CHECK(sep.long_range_density_fourier(sep.long_range_density_fourier_radius()) == -1e-15_a);

		// the G = 0 component must be -1.0
		CHECK(sep.long_range_density_fourier(0.0) == -1.0_a);
		
		// the integral must be zero component of the real space density
		double dg = 0.1;
		double integral = 0.0;
		for(double g = dg; g < sep.long_range_density_fourier_radius(); g += dg){
			integral += dg*4.0*M_PI*g*g*sep.long_range_density_fourier(g);
		}
		
		CHECK(integral/pow(2*M_PI, 3) == Approx(sep.long_range_density(0.0)));
	}
	
}

#endif

#endif

