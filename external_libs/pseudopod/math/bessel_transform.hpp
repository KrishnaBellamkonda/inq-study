/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDOPOD__MATH__BESSEL_TRANSFORM
#define PSEUDOPOD__MATH__BESSEL_TRANSFORM

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "spherical_bessel.hpp"
#include <gpu/run.hpp>

namespace pseudo {
namespace math {

class bessel_transform {
public:
	bessel_transform(double const gmax, int size):
		size_(size),
		dg_(gmax/(size_ - 1)){
	}

	auto dg() const{
		return dg_;
	}

	template <class VectorType>
	VectorType forward(int const ll, VectorType const & grid, VectorType const & potential) const {
		VectorType transform(size_);

		gpu::run(size_, [ll, pot = begin(potential), tra = begin(transform), potsize = potential.size(), gr = begin(grid), dg = dg_] GPU_LAMBDA (auto ig){
			auto gg = dg*ig;
			
			double integral = 0.0;
			for(unsigned ip = 1; ip < potsize - 1; ip++){
				double dr = 0.5*(gr[ip + 1] - gr[ip - 1]);
				double rr = gr[ip];
				integral += dr*rr*rr*pot[ip]*spherical_bessel(ll, rr*gg);
			}
				
			tra[ig] = sqrt(2.0/M_PI)*integral;
		});
		
		return transform;				
	}

	template <class VectorType>
	void backward(int const ll, VectorType & potential_fs, VectorType const & grid, VectorType & potential) const {

		gpu::run(potential.size() - 1, [ll, pfs = begin(potential_fs), gr = begin(grid), pot = begin(potential), size = size_, dg = dg_] GPU_LAMBDA (auto ip){
			
			double rr = gr[ip];
					
			double integral = 0.0;
			for(int ig = 0; ig < size; ig++){
				auto gg = dg*ig;
				integral += dg*gg*gg*pfs[ig]*spherical_bessel(ll, rr*gg);
			}
			
			pot[ip] = sqrt(2.0/M_PI)*integral;
		});
		
	}
			
private:

	int size_;
	double dg_;
			
};
}
}

#ifdef PSEUDOPOD_UNIT_TEST
#include <catch2/catch.hpp>

TEST_CASE("class pseudo::math::bessel_transform", "[pseudo::math::bessel_transform]") {
	
	using namespace Catch::literals;

	double dr = 0.0137;
	int nr = 897;
	
	pseudo::math::bessel_transform trans(8.0, 1000);

	std::vector<double> grid(nr);
	std::vector<double> potential(nr);

	for(int ir = 0; ir < nr; ir++){
		double rr = ir*dr;
		grid[ir] = ir*dr;
		potential[ir] = exp(-rr*rr);
	}

	auto potential_fs = trans.forward(0, grid, potential);
	
	double diff_fs = 0.0;
	for(unsigned ig = 0; ig < potential_fs.size(); ig++){
		double gg = ig*trans.dg();

		//this constant is set by hand for now
		auto gauss = 0.353553*exp(-0.25*gg*gg);
		
		//std::cout << potential_fs[ig] << '\t' << gauss << '\t' << potential_fs[ig]/gauss << '\t' << log(potential_fs[ig])/log(gauss) << std::endl;
		
		diff_fs += fabs(potential_fs[ig] - gauss);
	}

	CHECK(fabs(diff_fs) < 1e-4);

	double scale = 0.777;
	
	for(unsigned ig = 0; ig < potential_fs.size(); ig++) potential_fs[ig] *= scale;
		
	trans.backward(0, potential_fs, grid, potential);

	double diff = 0.0;
	for(int ir = 0; ir < nr; ir++){
		double rr = ir*dr;
		diff += fabs(potential[ir] - scale*exp(-rr*rr));
	}

	CHECK(fabs(diff) < 1e-4);
	
}
#endif
 
#endif

#ifdef INQ_MATH_BESSEL_TRANSFORM_UNIT_TEST
#undef INQ_MATH_BESSEL_TRANSFORM_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
