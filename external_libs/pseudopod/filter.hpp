/* -*- indent-tabs-mode: t -*- */

#ifndef FILTER_HPP
#define FILTER_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "math/spline.hpp"
#include "math/bessel_transform.hpp"

#include <kalimotxo/cali.h>
#include <gpu/run.hpp>

namespace pseudo {
  
class filter {

public:

	filter(){
      
		std::vector<double> mask_x;
		std::vector<double> mask_f;

		auto file = std::ifstream(pseudopod::path::share() + "/filter/mask_function.data");
		assert(file);

		while(true){
			double xx, ff;
			file >> xx >> ff;

			if(file.eof()) break;

			mask_x.push_back(xx);
			mask_f.push_back(ff);
		}

		mask_function_ = pseudo::math::spline(mask_x.data(), mask_f.data(), mask_f.size(), pseudo::math::spline::boundary_condition::natural, pseudo::math::spline::boundary_condition::natural);
	}

	auto mask_function(double x) const {
		return mask_function_.function()(x);
	}
    
    
	template <class VectorType>
	void operator()(int const ll, double const gcutoff, VectorType const & grid, VectorType & potential) const{
			
		CALI_CXX_MARK_SCOPE("pseudopod::filter");
			
		assert(grid.size() == potential.size());

		double const alpha = 1.1;
		double const gamma = 2.0;

		auto imax = cutoff_i(potential, 0.001);

		auto rcut = gamma*grid[imax];

		// Apply a gaussian cutoff to the tail of the function
		gpu::run(potential.size(),	[rcut, gr = begin(grid), pot = begin(potential), potsize = potential.size()] GPU_LAMBDA (auto ii){
			auto tailcut = 0.6*rcut;
			if(gr[ii] >= tailcut){
				auto sigma = (gr[potsize - 1] - tailcut)/6.0;
				pot[ii] *= exp(-pow((gr[ii] - tailcut)/sigma, 2));
			}
 		});

		auto mask1 = mask_function_.function()(1.0);
		
		for(unsigned ii = 0; ii < potential.size(); ii++) {
			if(grid[ii] <= rcut) {
				potential[ii] /= mask_function_.function()(grid[ii]/rcut);
			} else {
				potential[ii] *= (1.0 - (grid[ii] - rcut)/(grid[potential.size() - 1] - rcut))/mask1;
			}
		}

		math::bessel_transform transform(4.0*gcutoff, 1000);

		auto potential_fs = transform.forward(ll, grid, potential);

		auto beta = log(1.0e5)/pow(alpha - 1.0, 2);
		auto effcut = gcutoff/alpha;
			
		gpu::run(potential_fs.size(), [dg = transform.dg(), effcut, beta, pfs = begin(potential_fs)] GPU_LAMBDA (auto ig){
			auto gg = ig*dg;
			if(gg >= effcut) {
				auto exp_arg = -beta*pow(gg/effcut - 1.0, 2);
				if(exp_arg > -100.0){
					pfs[ig] *= exp(exp_arg);
				} else {
					pfs[ig] = 0.0;
				}
			}
		});
			
		transform.backward(ll, potential_fs, grid, potential);
			
		for(unsigned ii = 0; ii < potential.size(); ii++) {
			if(grid[ii] <= rcut) {
				potential[ii] *= mask_function_.function()(grid[ii]/rcut);
			} else {
				potential[ii] *= mask1;					
			}
		}
      
	}

private:

	template <class VectorType>
	auto cutoff_i(VectorType & potential, double threshold) const {
		for(int ip = potential.size() - 1; ip >= 0; ip--){
			if(fabs(potential[ip]) >= threshold) return ip;
		}
		return 0;
	}
    
	pseudo::math::spline mask_function_;

};
  
}

#ifdef INQ__FILTER_UNIT_TEST
#undef INQ__FILTER_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <gpu/array.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
	
  pseudo::filter filter;

  SECTION("Mask function"){
    //values on the grid
    CHECK(filter.mask_function(0.0) == 0.10000000E+01_a);
    CHECK(filter.mask_function(0.1) == 0.93383589E+00_a);
    CHECK(filter.mask_function(0.2) == 0.75859548E+00_a);
    CHECK(filter.mask_function(0.3) == 0.53286431E+00_a);
    CHECK(filter.mask_function(0.4) == 0.31999705E+00_a);
    CHECK(filter.mask_function(0.5) == 0.16123200E+00_a);
    CHECK(filter.mask_function(0.6) == 0.66108380E-01_a);
    CHECK(filter.mask_function(0.7) == 0.20934765E-01_a);
    CHECK(filter.mask_function(0.8) == 0.46327998E-02_a);
    CHECK(filter.mask_function(0.9) == 0.56169212E-03_a);
    CHECK(filter.mask_function(1.0) == 0.98138215E-05_a);

    //interpolated values
    CHECK(filter.mask_function(0.002) == 1.000039716_a);
    CHECK(filter.mask_function(0.203) == 0.7522211149_a);
    CHECK(filter.mask_function(0.531) == 0.1252587238_a);
    CHECK(filter.mask_function(0.616) == 0.0560636886_a);
    CHECK(filter.mask_function(0.876) == 0.0010111218_a);    

  }

	SECTION("Filter potential"){

		std::vector<double> grid;
		std::vector<double> potential; 

		{

			std::cout << pseudopod::path::unit_tests_data() + "/filter_test_function.dat" << std::endl;
			auto file = std::ifstream(pseudopod::path::unit_tests_data() + "/filter_test_function.dat");
			assert(file);
		 
			while(true){
				double xx, ff;
				file >> xx >> ff;
			 
				if(file.eof()) break;
			 
				grid.push_back(xx);
				potential.push_back(ff);
			}
		}

		pseudo::filter filter;

		auto potential_copy = potential;

		gpu::array<double, 1> grid_gpu(grid.begin(), grid.end());
		gpu::array<double, 1> potential_gpu(potential.begin(), potential.end());
			
		filter(0, 10.471975511965978, grid_gpu, potential_gpu);

		/*
			std::ofstream file("filter.dat");
	 
			for(unsigned ii = 0; ii < potential.size(); ii++){
			file << grid[ii] << '\t' << potential_copy[ii] << '\t' << potential[ii] << std::endl;
			}
		*/

		CHECK(potential_gpu[0]   == 4.8010420369_a);
		CHECK(potential_gpu[10]  == 4.7380549687_a);
		CHECK(potential_gpu[20]  == 4.5482932165_a);
		CHECK(potential_gpu[30]  == 4.2381984034_a);
		CHECK(potential_gpu[40]  == 3.8191241471_a);
		CHECK(potential_gpu[50]  == 3.3110188587_a);
		CHECK(potential_gpu[60]  == 2.7449714685_a);
		CHECK(potential_gpu[70]  == 2.1630510999_a);
		CHECK(potential_gpu[80]  == 1.6133438068_a);
		CHECK(potential_gpu[90]  == 1.1403313163_a);
		CHECK(potential_gpu[100] == 0.7738077306_a);
		CHECK(potential_gpu[120] == 0.3677118499_a);
		CHECK(potential_gpu[140] == 0.2339165487_a);
		CHECK(potential_gpu[160] == 0.1512647623_a);
		CHECK(potential_gpu[180] == 0.0569584706_a);
		CHECK(potential_gpu[200] == -0.0071591493_a);
		CHECK(potential_gpu[220] == -0.0229415535_a);
		CHECK(potential_gpu[240] == -0.0134239856_a);
		CHECK(potential_gpu[260] == -0.0031501591_a);
		CHECK(potential_gpu[280] ==  0.0001769939_a);
		CHECK(potential_gpu[300] == -0.0000223473_a);
		CHECK(potential_gpu[320] == -0.0001335133_a);
		CHECK(potential_gpu[340] ==  0.0000343243_a);
		CHECK(potential_gpu[360] ==  0.0000146211_a);
		CHECK(potential_gpu[380] == -0.0000198203_a);
		
  }
	
}
#endif
 
#endif
