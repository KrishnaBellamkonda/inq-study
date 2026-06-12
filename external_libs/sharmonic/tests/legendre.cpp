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

auto compare() {
#ifdef HAVE_SPH_LEGENDRE
	auto const tol = 2e-13;
	auto const ctol = 2e-12; //the cosine version has lower precision due to the cos to sin calculation
	auto reps = 1000;
	
	for(int ll = 0; ll <= 12; ll++) {
		for(int mm = -ll; mm <= ll; mm++) {
			
			srand48(0xFC769E60887BB49Cl);

			for(int iter = 0; iter < reps; iter++){
				double theta = M_PI*drand48();

				auto ref = std::sph_legendre(ll, abs(mm), theta);
				auto leg = sharmonic::angular_legendre(ll, mm, theta);

				auto diff = fabs(ref - leg);
				if(diff > tol) std::cout << ll << '\t' << mm << '\t' << theta << '\t' << ref << '\t' << '\t' << leg << '\t' << diff << std::endl;
				CHECK(diff < tol);

				auto cleg = sharmonic::cos_legendre(ll, mm, cos(theta));
				auto cdiff = fabs(ref - cleg);
				if(cdiff > ctol) std::cout << ll << '\t' << mm << '\t' << theta << '\t' << ref << '\t' << '\t' << cleg << '\t' << cdiff << std::endl;
				CHECK(cdiff < ctol);
				
			}

		}
	}
#endif
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
