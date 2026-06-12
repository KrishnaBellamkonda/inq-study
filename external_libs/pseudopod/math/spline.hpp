/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef SPLINE_H
#define SPLINE_H

#include <vector>
#include <cmath>

#include <gpu/array.hpp>
#include <gpu/run.hpp>

namespace pseudo {
namespace math {

class spline {
	
public:

	using container_type = gpu::array<double, 1>;

	enum class boundary_condition { flat,   // zero first derivative
																	natural // zero second derivative
	};
	
	spline(){
	}

	template <class GridType, class ValuesType>
	spline(GridType const & grid, ValuesType const & values, int size, boundary_condition low_bc, boundary_condition high_bc):
		grid_(size),
		func_(size),
		der2_(size)	{
		
		for(int ii = 0; ii < size; ii++){
			grid_[ii] = grid[ii];
			func_[ii] = values[ii];
			assert(ii == 0 or grid_[ii] > grid_[ii - 1]);
		}

		//calculate the second derivative array der2_
		auto tmp = container_type(size);

		if(low_bc == boundary_condition::flat){
			der2_[0] = -0.5;
			tmp[0] = (3.0/(grid_[1] - grid_[0]))*((func_[1] - func_[0])/(grid_[1] - grid_[0]));
		} else {
			der2_[0] = 0.0;
			tmp[0] = 0.0;
		}
		
		for(int ii = 1; ii < size - 1; ii++) {
			auto sig = (grid_[ii] - grid_[ii - 1])/(grid_[ii + 1] - grid_[ii - 1]);
			auto pp = sig*der2_[ii - 1] + 2.0;
			der2_[ii] = (sig - 1.0)/pp;
			tmp[ii] = (6.0*((func_[ii + 1] - func_[ii])/(grid_[ii + 1] - grid_[ii]) - (func_[ii] - func_[ii - 1])/(grid_[ii] - grid_[ii - 1]))/(grid_[ii + 1] - grid_[ii - 1]) - sig*tmp[ii - 1])/pp;
		}

		if(high_bc == boundary_condition::flat){
			auto un = (3.0/(grid_[size - 1] - grid_[size - 2]))*(-(func_[size - 1] - func_[size - 2])/(grid_[size - 1] - grid_[size - 2]));
			der2_[size - 1] = (un - 0.5*tmp[size - 2])/(0.5*der2_[size - 2] + 1.0);
		} else {
			der2_[size - 1] = 0.0;
		}
		
		for(int ii = size - 2; ii >= 0; ii--) der2_[ii] = der2_[ii]*der2_[ii + 1] + tmp[ii];
	}
	
	double cutoff_radius(double threshold) const {
		for(int ip = func_.size() - 1; ip >= 0; ip--) if(fabs(func_[ip]) >= threshold) return grid_[ip];
		return 0.0;
	}

	long size() const {
		return grid_.size();
	}	

	class const_iterator {

		struct search_result {
			int low;
			int high;
			double hh;
			double aa;
			double bb;

			GPU_FUNCTION auto value(typename container_type::const_iterator func, typename container_type::const_iterator der2){
				return aa*func[low] + bb*func[high] + hh*hh*1.0/6.0*(aa*(aa*aa - 1.0)*der2[low] + bb*(bb*bb - 1.0)*der2[high]);
			}

			GPU_FUNCTION auto derivative(typename container_type::const_iterator func, typename container_type::const_iterator der2){
				return (func[high] - func[low])/hh + hh*((1.0/6.0 - 0.5*aa*aa)*der2[low] + (0.5*bb*bb - 1.0/6.0)*der2[high]);
			}
		};
		
		GPU_FUNCTION auto search_position(const double & point) const {
			assert(point >= grid_[0] and point <= grid_[size_ - 1]);

			search_result ret;
			
			ret.low = 0;
			ret.high = size_ - 1;
			
			while (ret.high - ret.low > 1) {
				auto mid = (ret.high + ret.low)/2;
				if (grid_[mid] > point) {
					ret.high = mid;
				} else {
					ret.low = mid;
				}
			}
			assert(ret.high - ret.low == 1);

			ret.hh = grid_[ret.high] - grid_[ret.low];
			assert(ret.hh > 0.0);

			ret.aa = (grid_[ret.high] - point)/ret.hh;
			ret.bb = (point - grid_[ret.low])/ret.hh;

			return ret;
		}
		
	public:

		GPU_FUNCTION auto operator()(const double & point) const {
			auto pos = search_position(point);
			return pos.value(func_, der2_);
		}

		GPU_FUNCTION void derivative(const double & point, double & val, double & der) const {
			auto pos = search_position(point);
			val = pos.value(func_, der2_);
			der = pos.derivative(func_, der2_);
		}
		
		GPU_FUNCTION double derivative(const double & point) const {
			auto pos = search_position(point);
			return pos.derivative(func_, der2_);
		}

		friend class spline;
		
	private:

		const_iterator(typename container_type::const_iterator arg_x, typename container_type::const_iterator arg_y, typename container_type::const_iterator arg_y2, long arg_size)
			:grid_(arg_x),
			 func_(arg_y),
			 der2_(arg_y2),
			 size_(arg_size)
		{
		}

		typename container_type::const_iterator grid_;
		typename container_type::const_iterator func_;
		typename container_type::const_iterator der2_;
		long size_;

	};

	auto function() const {
		return const_iterator(grid_.cbegin(), func_.cbegin(), der2_.cbegin(), grid_.size());
	}
	
private :
    
	container_type grid_;
	container_type func_;
	container_type der2_;
	
};

}
}
#endif


#ifdef INQ_MATH_SPLINE_UNIT_TEST
#undef INQ_MATH_SPLINE_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <gpu/run.hpp>
#include <gpu/array.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;

  const int nn = 20;
  const double dx = 0.5;
  std::vector<double> xx(nn);
  std::vector<double> ff(nn);

  for(int ii = 0; ii < nn; ii++){
    xx[ii] = dx*ii;
    ff[ii] = cos(xx[ii]);
  }

  pseudo::math::spline spl(xx.data(), ff.data(), nn, pseudo::math::spline::boundary_condition::flat, pseudo::math::spline::boundary_condition::natural);

  SECTION("Check single value interpolation"){

    CHECK(spl.function()(0.0) == 1.0_a);
    CHECK(spl.function()(0.4) == 0.9209862343_a);
    CHECK(spl.function()(M_PI/2.0) == 0.0000212848_a);
    CHECK(spl.function()(M_PI) == -0.9998862784_a);

  }

  SECTION("Check single value derivatives"){

    CHECK(spl.function().derivative(0.0) == Approx(0.0).margin(1e-8));
    CHECK(spl.function().derivative(0.4) == -0.3884027755_a);
    CHECK(spl.function().derivative(M_PI/2.0) == -0.9997597149_a);
    CHECK(spl.function().derivative(M_PI) == 0.0009543059_a);

  }
  
  SECTION("Check single value and derivative"){

    double f1, df1;

    spl.function().derivative(3.0/4.0*M_PI, f1, df1);
    
    CHECK(f1 == -0.7070426574_a);
    CHECK(df1 == -0.7077576301_a);

    spl.function().derivative(8.3, f1, df1);
    
    CHECK(f1 == -0.4304260777_a);
    CHECK(df1 == -0.9021271854_a);

  }
 
  SECTION("Spline in gpu::run"){
		
		using gpu_array = gpu::array<double, 1>;
		
		auto value = gpu_array(1);
		
		gpu::run([val = begin(value), spline_function = spl.function()] GPU_LAMBDA (){
			val[0] = spline_function(0.4);
		});

		gpu::sync();
		
		CHECK(value[0] == 0.9209862343_a);
	}

}

#endif
