/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_PAWXML_HPP
#define PSEUDO_PAWXML_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <fstream>
#include <vector>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>

#include "base.hpp"
#include "../element.hpp"

extern "C"
{
	void prep_read_pawxml_(char*); 
	//					   filename 
	extern int pawrad_mesh_size_(char*); 
	//					         filename
	extern int pawtab_mesh_size_(char*); 
	//					         filename
	extern double pawxml_zval_(char*); 
	//					       filename
	extern int pawxml_lmax_(char*); 
	//					    filename
	extern int pawproj_exist_(char*,int&,int&); 
	//					      filename, l, ichan
	extern int pawxml_nchan_max_(char*); 
	//					         filename
	extern int pawxml_nproj_(char*);
    //                           filename
	void pawrad_grid_(char*,int&,double*);
    //               filename,mesh_size,radial_grid
	void pawxml_local_pp_(char*,int&,double*);
    //                    filename,mesh_size,loc_pot
	void pawproj_(char*,int&,int&,int&,double*);
    //            filename, l, ichan, mesh_size, projector
	void get_sij_(char*,int&,double*);
    //            filename, lmn_size, sij
}
namespace pseudo {

class pawxml : public pseudo::base {

public:

	pawxml(const std::string & filename){

		filename_ = filename;
		type_ = pseudo::type::PAW;


		c_filename_ = new char[264];
		std::strcpy(c_filename_,filename.c_str());
		prep_read_pawxml_(c_filename_);
		valence_charge_ = pawxml_zval_(c_filename_);
		nchannels_      = pawxml_nchan_max_(c_filename_);
		nprojectors_lm_ = pawxml_nproj_(c_filename_);
		lmax_           = pawxml_lmax_(c_filename_);
		mesh_size_      = pawtab_mesh_size_(c_filename_);
		mesh_size_rad_  = pawrad_mesh_size_(c_filename_);
		grid(grid_);
		nlcc_           = false;
		has_density_    = false; 
		/*
			For now, we will rely on host cost to provide initial guess
			for atomic valence density (tnval).  Extracting tnval seems 
			to be complicated in libpaw/Abinit because of potential mesh 
			shifts and the most immediately available version of tnval 
			in pawtab is in reciprocal space for some reason.  
			Future todo: Either transform this into real-space or take
			the real-space tnval from pawsetup object?
			Refer to libpaw/pawpsp.f90 line 3745 for details.
		*/
		llocal_         = -1;

		int m_size = mesh_size();
		local_potential_.resize(m_size);
		pawxml_local_pp_(c_filename_,m_size,local_potential_.data());

		sij_.resize(nprojectors_lm_, std::vector<double>(nprojectors_lm_, 0.0));
		std::vector<double> sij_flatten_(nprojectors_lm_ * nprojectors_lm_);
		get_sij_(c_filename_, nprojectors_lm_, sij_flatten_.data());

		// Convert column-major (sij_flatten_) to row-major (sij_)
		for (int row = 0; row < nprojectors_lm_; ++row) {
			for (int col = 0; col < nprojectors_lm_; ++col) {
				sij_[row][col] = sij_flatten_[row + col * nprojectors_lm_];
			}
		}
	}

	pseudo::format format() const { return pseudo::format::PAWXML; }

	void grid(std::vector<double> & val) const {

	    std::vector<double> pawrad_grid;
		int m_size = pawrad_mesh_size_(c_filename_);
		pawrad_grid.resize(m_size);
		pawrad_grid_(c_filename_,m_size,pawrad_grid.data());

		val.resize(mesh_size());
		std::copy(pawrad_grid.begin(), pawrad_grid.begin() + mesh_size_, val.begin());
	}
    
	int size() const {
		return file_size_;
	};

	bool has_projector(int l, int i) const {
		i+=1; //Fortran indexes from 1
		return pawproj_exist_(c_filename_, l, i);
	}

	std::string description() const {
		return description_;
	}
    
	std::string symbol() const {
		pseudo::element el(atomic_number_);
		return el.symbol();
	}

	int atomic_number() const {
		return atomic_number_;
	}

	double mass() const {
		pseudo::element el(atomic_number_);
		return el.mass();
	}
    
	double valence_charge() const {
		return valence_charge_;
	}

	pseudo::exchange exchange() const {
		if(ixc_ > 0){
			if(ixc_ == 1) return pseudo::exchange::NONE;
			if(ixc_ >= 2 && ixc_ <= 9) return pseudo::exchange::LDA;
			if(ixc_ == 11 || ixc_ == 12) return pseudo::exchange::PBE;
		}	else {
			return pseudo::exchange((-ixc_ +ixc_%1000)/1000);
		}
      
		return pseudo::exchange::UNKNOWN;
    
	}

	pseudo::correlation correlation() const {
		if(ixc_ > 0){
			if(ixc_ == 1) return pseudo::correlation::LDA_XC_TETER93;
			if(ixc_ == 2) return pseudo::correlation::LDA_PZ;
			if(ixc_ == 7) return pseudo::correlation::LDA_PW;
			if(ixc_ == 7 || ixc_ == 8) return pseudo::correlation::NONE;
			if(ixc_ == 11) return pseudo::correlation::PBE;
			if(ixc_ == 12) return pseudo::correlation::NONE;
		}	else {
			return pseudo::correlation(-ixc_%1000);
		}
      
		return pseudo::correlation::UNKNOWN;

	}

	int llocal() const {
		if(llocal_ > lmax_) return -1;
		return llocal_;
	}

	int nchannels() const {
		return nchannels_;
	}
    
	double mesh_spacing() const {
		return mesh_spacing_;
	}

	int mesh_size() const {
		return mesh_size_;
	}
    
	void local_potential(std::vector<double> & potential) const {
		potential.resize(mesh_size_);
		assert(mesh_size_ == int(local_potential_.size()));
		for(int ip = 0; ip < mesh_size_; ip++) potential[ip] = local_potential_[ip];
	}

	int nprojectors() const {
		return nprojectors_;
	}
    
	void projector(int l, int i, std::vector<double> & proj) const {
		i+=1; //Fortran indexes from 1
		proj.clear();
		proj.resize(mesh_size_);
		int m_size = mesh_size();
		pawproj_(c_filename_,l,i,m_size,proj.data());

		//the projectors from libPAW are multiplied by r, so we have to divide and fix the first point
		assert(proj[0] == 0.0 && "first point of proj is not zero as expected");
		for(unsigned ii = 1; ii < proj.size(); ii++) proj[ii] /= grid_[ii];
		extrapolate_first_point(proj);
	}

	// PAW uses dynamic D_ij that depends on pseudo wavefunctions so need to calculate D_ij elsewhere
	double d_ij(int l, int i, int j) const {
		if (i==j) {
			return 1.0;
		} else {
			return 0.0;
		}
	}

	double s_ij(int i, int j) const {
		return sij_[i][j];
	}

	bool has_radial_function(int/*l*/) const{
		return false;
	}

	void radial_function(int/*l*/, std::vector<double> & function) const {
		function.clear();
	}

	void radial_potential(int/*l*/, std::vector<double> & function) const {
		function.clear();
	}

	bool has_nlcc() const{
		return nlcc_;
	}

	bool has_overlap() const{
		return true;
	}
    
	bool has_density() const{
		return has_density_;
	}

	void density(std::vector<double> & density) const {
		density.resize(mesh_size_);
		assert(mesh_size_ == int(density_.size()));
		for(int ip = 0; ip < mesh_size_; ip++) density[ip] = density_[ip]/(4.0*M_PI);
	}
    
private:
    
	void extrapolate_first_point(std::vector<double> & function_) const{

		assert(function_.size() >= 4);

		double x1 = grid_[1];
		double x2 = grid_[2];
		double x3 = grid_[3];
		double f1 = function_[1];
		double f2 = function_[2];
		double f3 = function_[3];


		// obtained from:
		// http://www.wolframalpha.com/input/?i=solve+%7Bb*x1%5E2+%2B+c*x1+%2B+d+%3D%3D+f1,++b*x2%5E2+%2B+c*x2+%2B+d+%3D%3D+f2,+b*x3%5E2+%2B+c*x3+%2B+d+%3D%3D+f3+%7D++for+b,+c,+d
      
		function_[0] = f1*x2*x3*(x2 - x3) + f2*x1*x3*(x3 - x1) + f3*x1*x2*(x1 - x2);
		function_[0] /= (x1 - x2)*(x1 - x3)*(x2 - x3);

	}
    
	size_t file_size_;
	std::string description_;
	int atomic_number_;
	double valence_charge_;
	int ixc_;
	int llocal_;
	int mesh_size_;
	int mesh_size_rad_;
	int nchannels_;
	double mesh_spacing_;
	std::vector<int> nprojl_;
	int nprojectors_;
	int nprojectors_lm_;
	std::vector<std::vector<std::vector<double> > > projectors_;
	std::vector<std::vector<double> > ekb_;
	std::vector<std::vector<double> > sij_;
	std::vector<double> local_potential_;
	std::vector<double> grid_;
	bool nlcc_;
	std::vector<double> nlcc_density_;
	bool has_density_;
	std::vector<double> density_;
	char* c_filename_;
    
};

}

#endif

#ifdef INQ_FORMATS_PAWXML_UNIT_TEST
#undef INQ_FORMATS_PAWXML_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
}
#endif
