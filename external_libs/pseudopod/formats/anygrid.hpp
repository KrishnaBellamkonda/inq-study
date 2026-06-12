/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_ANYGRID_HPP
#define PSEUDO_ANYGRID_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <vector>
#include <string>
#include "../math/spline.hpp"
#include "base.hpp"

namespace pseudo {

class anygrid : public pseudo::base {

public:

	anygrid(bool uniform_grid):
		uniform_grid_(uniform_grid){
	}
    
	double mesh_spacing() const {
		return 0.01;
	}
    
	int mesh_size() const {
		if(uniform_grid_) {
			return mesh_size_;
		} else {
			return grid_.size();
		}
	}
    
	virtual void grid(std::vector<double> & val) const {
		if(uniform_grid_){
			pseudo::base::grid(val);
		} else {
			val = grid_;
		}
	}

	virtual void grid_weights(std::vector<double> & val) const {
		if(uniform_grid_){
			pseudo::base::grid(val);
		} else {
			val = grid_weights_;
		}
	}

protected:

	void interpolate(std::vector<double> & function) const {
		if(!uniform_grid_) return;

		std::vector<double> function_in_grid = function;
      
		assert(function.size() == grid_.size());
      
		math::spline function_spline(grid_.data(), function_in_grid.data(), function_in_grid.size(), math::spline::boundary_condition::flat, math::spline::boundary_condition::natural);
		
		function.clear();
		for(double rr = 0.0; rr <= grid_[grid_.size() - 1]; rr += mesh_spacing()){
			function.push_back(function_spline.function()(rr));
		}
	}

	bool uniform_grid_;
	std::vector<double> grid_;
	std::vector<double> grid_weights_;
	int mesh_size_;
    
};

}

#endif

#ifdef INQ_FORMATS_ANYGRID_UNIT_TEST
#undef INQ_FORMATS_ANYGRID_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
