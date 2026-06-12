/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDOPOTENTIAL_HPP
#define PSEUDOPOTENTIAL_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "formats/psml.hpp"
#include "formats/qso.hpp"
#include "formats/upf1.hpp"
#include "formats/upf2.hpp"
#include "formats/psp8.hpp"
#include "formats/pawxml.hpp"
#include "detect_format.hpp"
#include "math/spline.hpp"
#include "erf_range_separation.hpp"
#include "filter.hpp"

#include <gpu/array.hpp>


namespace pseudo {

template <typename ContainerType = gpu::array<double, 1>>
class pseudopotential {

public:

	using spline_type = pseudo::math::spline;

private:

	bool has_total_angular_momentum_;
	bool has_overlap_;
	ContainerType grid_;
	spline_type short_range_;
	spline_type electronic_density_;
	spline_type nlcc_density_;
	double valence_charge_;
	std::vector<spline_type> projectors_;
	std::vector<int> projectors_l_;
	std::vector<double> projectors_2j_;
	std::vector<std::vector<double>> kb_coeff_;
	std::vector<std::vector<double>> overlap_mat_;
	int nproj_lm_;
	double short_range_potential_radius_;
	std::vector<int> channel_;
	
public:
	
	pseudopotential(const std::string & filename, const pseudo::math::erf_range_separation & sep, double gcutoff, bool apply_filter = true){

		CALI_CXX_MARK_SCOPE("pseudopod::pseudopotential");

		double const threshold = 1e-5;
		
		//PARSE THE FILE
		base * pseudo_file;    
		pseudo::format format = pseudo::detect_format(filename);
      
		if(format == pseudo::format::FILE_NOT_FOUND) throw std::runtime_error{"pseudopod error: file '" + filename + "' not found."};
		if(format == pseudo::format::UNKNOWN) throw std::runtime_error{"pseudopod error: unknown format for file '" + filename + "'."};
      
		switch(format){
		case pseudo::format::QSO:
			pseudo_file = new pseudo::qso(filename);
			break;
		case pseudo::format::UPF1:
			pseudo_file = new pseudo::upf1(filename);
			break;
		case pseudo::format::UPF2:
			pseudo_file = new pseudo::upf2(filename);
			break;
		case pseudo::format::PSML:
			pseudo_file = new pseudo::psml(filename);
			break;
		case pseudo::format::PSP8:
			pseudo_file = new pseudo::psp8(filename);
			break;
		case pseudo::format::PAWXML:
			pseudo_file = new pseudo::pawxml(filename);
			break;
		default:
			throw std::runtime_error{"pseudopod error: unsupported format for file '" + filename + "'."};
		}

		if(pseudo_file->type() != pseudo::type::KLEINMAN_BYLANDER && pseudo_file->type() != pseudo::type::PAW) {
			delete pseudo_file;
			throw std::runtime_error{"pseudopod error: unsupported pseudopotential type for file '" + filename + "'."};
		}

		has_total_angular_momentum_ = pseudo_file->has_total_angular_momentum();
		has_overlap_= pseudo_file->has_overlap();
		
		pseudo::filter filter;

		std::vector<double> grid_read;
		pseudo_file->grid(grid_read);

		grid_ = ContainerType(grid_read.size());
		for(unsigned ii = 0; ii < grid_.size(); ii++){
			grid_[ii] = grid_read[ii];
		}

		valence_charge_ = pseudo_file->valence_charge();

		//SEPARATE THE LOCAL PART
		std::vector<double> local_potential_read;
		pseudo_file->local_potential(local_potential_read);

		ContainerType local_potential(local_potential_read.size());

		for(unsigned ii = 0; ii < local_potential.size(); ii++){
			local_potential[ii] = local_potential_read[ii] - valence_charge_*sep.long_range_potential(grid_[ii]);
		}

		if(apply_filter) filter(0, gcutoff, grid_, local_potential);
		
		short_range_ = spline_type(grid_.data(), local_potential.data(), local_potential.size(), spline_type::boundary_condition::flat, spline_type::boundary_condition::natural);

		/*
			Precision in energy seems to be 1 to 2 orders of magnitude
			lower than the threshold. 1e-5 gives around 1e-7 error
			in the energy.
			
			Since this value does not affect the cost of the application
			of the hamiltonian we can use a small threshold.
			
			It makes the calculation of the local potential more expensive, so we
			should tweak if that becomes a bottleneck.
			
		*/
		short_range_potential_radius_ = short_range_.cutoff_radius(threshold);

		//THE PROJECTORS

		std::vector<double> proj_read;
		
		nproj_lm_ = 0;
		for(int ll = 0; ll <= pseudo_file->lmax(); ll++){
			for(int ichan = 0; ichan < pseudo_file->nchannels(); ichan++){
				if(ll == pseudo_file->llocal()) continue;
				if(not pseudo_file->has_projector(ll, ichan)) continue;
				
				pseudo_file->projector(ll, ichan, proj_read);

				if(proj_read.size() == 0) continue;

				ContainerType proj(proj_read.begin(), proj_read.end());
				
				assert(proj.size() <= grid_.size());
				
				if(apply_filter) filter(ll, gcutoff, grid_, proj);
					
				projectors_.emplace_back(grid_.data(), proj.data(), proj.size(), spline_type::boundary_condition::flat, spline_type::boundary_condition::natural);
				projectors_l_.push_back(ll);
				channel_.push_back(ichan);
				projectors_2j_.push_back(pseudo_file->projector_2j(ll, ichan));
				nproj_lm_ += 2*ll + 1;
			}
		}

		kb_coeff_.resize(projectors_.size(), std::vector<double>(projectors_.size(), 0.0));

		for(auto iproj = 0ul; iproj < projectors_.size(); iproj++){
			for(auto jproj = 0ul; jproj < projectors_.size(); jproj++){
				if(projectors_l_[iproj] != projectors_l_[jproj]) continue;
				kb_coeff_[iproj][jproj] = pseudo_file->d_ij(projectors_l_[iproj], channel_[iproj], channel_[jproj]);
			}
		}

		overlap_mat_.resize(nproj_lm_, std::vector<double>(nproj_lm_, 0.0));

		for(auto iproj = 0; iproj < nproj_lm_; iproj++){
			for(auto jproj = 0; jproj < nproj_lm_; jproj++){
				overlap_mat_[iproj][jproj] = pseudo_file->s_ij(iproj, jproj);
			}
		}

		//THE ELECTRONIC DENSITY
		if(pseudo_file->has_density()){
			std::vector<double> density;
			pseudo_file->density(density);
			electronic_density_ = spline_type(grid_.data(), density.data(), density.size(), spline_type::boundary_condition::flat, spline_type::boundary_condition::natural);
		}
			
		//THE NLCC DENSITY
		if(pseudo_file->has_nlcc()){
			std::vector<double> density;
			pseudo_file->nlcc_density(density);
			nlcc_density_ = spline_type(grid_.data(), density.data(), density.size(), spline_type::boundary_condition::flat, spline_type::boundary_condition::natural);
		}
			
		delete pseudo_file;
	}

	auto & has_total_angular_momentum() const {
		return has_total_angular_momentum_;
	}

	auto & has_overlap() const {
		return has_overlap_;
	}

	double overlap(int iproj, int jproj) const {
		if(iproj < 0 or unsigned(iproj) >= overlap_mat_.size()) throw std::runtime_error{"pseudopod error: projector index out of range overlap i"};
		if(jproj < 0 or unsigned(jproj) >= overlap_mat_.size()) throw std::runtime_error{"pseudopod error: projector index out of range overlap j"};
		return overlap_mat_[iproj][jproj];
	}
	
	
	const double & valence_charge() const {
		return valence_charge_;
	}
      
	const spline_type & short_range_potential() const {
		return short_range_;
	}

	double short_range_potential_radius() const {
		return short_range_potential_radius_;
	}
		
	double projector_radius() const {
		const double threshold = 0.001;
		double radius = 0.0;
		for(unsigned iproj = 0; iproj < projectors_.size(); iproj++){
			radius = std::max(radius, projectors_[iproj].cutoff_radius(threshold));
		}
		return radius;
	}

	// the number of projectors with different l
	int num_projectors_l() const {
		return projectors_.size();
	}

	//the number of projectors with different l and m
	int num_projectors_lm() const {
		return nproj_lm_;
	}

	const spline_type & projector(int iproj) const {
		if(iproj < 0 or unsigned(iproj) >= projectors_.size()) throw std::runtime_error{"pseudopod error: projector index out of range"};
		return projectors_[iproj];
	}

	int projector_l(int iproj) const {
		if(iproj < 0 or unsigned(iproj) >= projectors_l_.size()) throw std::runtime_error{"pseudopod error: projector index out of range"};
		return projectors_l_[iproj];
	}

	auto projector_2j(int iproj) const {
		if(iproj < 0 or unsigned(iproj) >= projectors_l_.size()) throw std::runtime_error{"pseudopod error: projector index out of range"};
		return projectors_2j_[iproj];
	}

	double kb_coeff(int iproj, int jproj) const {
		if(iproj < 0 or unsigned(iproj) >= kb_coeff_.size()) throw std::runtime_error{"pseudopod error: projector index out of range"};
		if(jproj < 0 or unsigned(jproj) >= kb_coeff_.size()) throw std::runtime_error{"pseudopod error: projector index out of range"};
		return kb_coeff_[iproj][jproj];
	}

	auto has_electronic_density() const {
		return electronic_density_.size() > 0;
	}
		
	const auto & electronic_density() const {
		return electronic_density_;
	}

	double electronic_density_radius() const {
		const double threshold = 1e-5;
		return electronic_density_.cutoff_radius(threshold);
	}

	auto has_nlcc_density() const {
		return nlcc_density_.size() > 0;
	}
		
	const auto & nlcc_density() const {
		return nlcc_density_;
	}

	double nlcc_density_radius() const {
		const double threshold = 1e-5;
		return nlcc_density_.cutoff_radius(threshold);
	}
    
};
  
}
#endif

#ifdef INQ__PSEUDOPOTENTIAL_UNIT_TEST
#undef INQ__PSEUDOPOTENTIAL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
	
	const pseudo::math::erf_range_separation sep(0.625);
	double gcutoff = 10.0;
	
  SECTION("Non-existing file"){
    CHECK_THROWS(pseudo::pseudopotential("/this_file_doesnt_exists", sep, gcutoff));
  }

  SECTION("Non-pseudopotential file"){
    CHECK_THROWS(pseudo::pseudopotential(pseudopod::path::unit_tests_data() + "benzene.xyz", sep, gcutoff));
  }

  SECTION("Non-supported format QSO/Qbox pseudopotential file"){
    CHECK_THROWS(pseudo::pseudopotential(pseudopod::path::unit_tests_data() + "I_HSCV_LDA-1.0.xml", sep, gcutoff));
  }

  SECTION("UPF2 pseudopotential file"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "W_ONCV_PBE-1.0.upf", sep, gcutoff);

    CHECK(ps.valence_charge() == 28.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);
    
    //values validated with Octopus
    CHECK(ps.valence_charge()*sep.long_range_potential(0.00000000E+00) == -3.57452283E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-04) == -3.57452282E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-02) == -3.57437033E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-02) == -3.57071367E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-01) == -3.55932992E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-01) == -3.22721954E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-00) == -2.49312397E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-00) == -5.60000000E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(6.01000000E-00) == -4.65890183E+00_a);

    CHECK(ps.valence_charge()*sep.long_range_density(0.00000000E+00) == -7.28195812E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-04) == -7.28195802E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-02) == -7.28102609E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-02) == -7.25869310E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-01) == -7.18934306E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-01) == -5.28778688E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-00) == -2.02465598E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-00) == -9.22199231E-14_a);
    CHECK(ps.valence_charge()*sep.long_range_density(6.01000000E-00) == -6.07009137E-20_a);

    CHECK(ps.short_range_potential_radius() == 3.64_a);
		
    CHECK(ps.short_range_potential().function()(0.00000000E+00) == 4.7640304436_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-02) == 4.7637432354_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-02) == 4.7503169936_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-01) == 4.7065468262_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-01) == 3.3182270188_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-00) == 0.774328497_a);

    CHECK(ps.num_projectors_l() == 8);
    CHECK(ps.num_projectors_lm() == 32);
    
    CHECK(ps.projector(0).function()(0.00000000E+00) == -6.1566565901_a);
    CHECK(ps.projector(0).function()(1.00000000E-02) == -6.1554764134_a);
    CHECK(ps.projector(0).function()(5.00000000E-02) == -6.1149296566_a);
    CHECK(ps.projector(0).function()(1.00000000E-01) == -5.9877899009_a);
    CHECK(ps.projector(0).function()(5.00000000E-01) == -2.7321598568_a);
    CHECK(ps.projector(0).function()(1.00000000E-00) ==  0.4769982542_a);
    CHECK(ps.projector(0).function()(6.01000000E-00) ==  0.0000000000_a);

    CHECK_THROWS(ps.projector(-1));
    CHECK_THROWS(ps.projector(8));

    CHECK(ps.projector_l(0) == 0);
    CHECK(ps.projector_l(1) == 0);
    CHECK(ps.projector_l(2) == 1);
    CHECK(ps.projector_l(3) == 1);
    CHECK(ps.projector_l(4) == 2);
    CHECK(ps.projector_l(5) == 2);
    CHECK(ps.projector_l(6) == 3);
    CHECK(ps.projector_l(7) == 3);

		CHECK(ps.kb_coeff(0, 0) == 8.243498986_a);
    CHECK(ps.kb_coeff(1, 1) == 0.9729808132_a);
    CHECK(ps.kb_coeff(2, 2) == 3.8116529178_a);
    CHECK(ps.kb_coeff(3, 3) == 0.4990369016_a);
    CHECK(ps.kb_coeff(4, 4) == 1.4609009471_a);
    CHECK(ps.kb_coeff(5, 5) == 0.2713485048_a);
    CHECK(ps.kb_coeff(6, 6) == -5.0048898015_a);
		CHECK(ps.kb_coeff(7, 7) == -2.9667606006_a);

		for(int ii = 0; ii < 8; ii++){
			for(int jj = 0; jj < 8; jj++){
				if(ii == jj) continue;
				CHECK(ps.kb_coeff(ii, jj) == 0.0_a);
			}
		}
		
    CHECK_THROWS(ps.projector_l(-100));    
    CHECK_THROWS(ps.projector_l(8));    
    
    CHECK(ps.projector_radius() == 2.88_a);

		CHECK(ps.has_electronic_density());

		CHECK(ps.electronic_density_radius() == 6.01_a);

		CHECK(not ps.has_nlcc_density());

		CHECK(not ps.has_overlap());
		
  }

  SECTION("UPF1 pseudopotential file"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "F.UPF", sep, gcutoff, false);

    CHECK(ps.valence_charge() == 7.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 2.8262095048_a);
		
    CHECK(ps.num_projectors_l() == 1);
    CHECK(ps.num_projectors_lm() == 1);
    
    CHECK(ps.projector_radius() == 0.8138477088_a);

		CHECK(ps.has_electronic_density());

		CHECK(ps.electronic_density_radius() == 5.0108166066_a);

		CHECK(not ps.has_overlap());
		
  }

  SECTION("PAWXML pseudopotential file"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "C_PAW.xml", sep, gcutoff);

    CHECK(ps.valence_charge() == 4.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 1.5507622759_a);
		
    CHECK(ps.num_projectors_l() == 4);
    CHECK(ps.num_projectors_lm() == 8);
    
    CHECK(ps.projector_radius() == 1.5419844767_a);

	CHECK(ps.has_overlap());

	CHECK(not ps.has_electronic_density());

	CHECK(not ps.has_nlcc_density());

	CHECK(ps.overlap(0, 0) == Approx(-5.8999736e-02));
	CHECK(ps.overlap(0, 1) == Approx( 7.4723649e-01));
	CHECK(ps.overlap(0, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(0, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(0, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(0, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(0, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(0, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(1, 0) == Approx( 7.4723649e-01));
	CHECK(ps.overlap(1, 1) == Approx(-9.3061095073));
	CHECK(ps.overlap(1, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(1, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(1, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(1, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(1, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(1, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(2, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(2, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(2, 2) == Approx( 6.9942573e-02));
	CHECK(ps.overlap(2, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(2, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(2, 5) == Approx(-4.8380555e-01));
	CHECK(ps.overlap(2, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(2, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(3, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(3, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(3, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(3, 3) == Approx( 6.9942573e-02));
	CHECK(ps.overlap(3, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(3, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(3, 6) == Approx(-4.8380555e-01));
	CHECK(ps.overlap(3, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(4, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 4) == Approx( 6.9942573e-02));
	CHECK(ps.overlap(4, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(4, 7) == Approx(-4.8380555e-01));

	CHECK(ps.overlap(5, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(5, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(5, 2) == Approx(-4.8380555e-01));
	CHECK(ps.overlap(5, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(5, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(5, 5) == Approx( 3.2034338050));
	CHECK(ps.overlap(5, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(5, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(6, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(6, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(6, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(6, 3) == Approx(-4.8380555e-01));
	CHECK(ps.overlap(6, 4) == Approx( 0.0000000000));
	CHECK(ps.overlap(6, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(6, 6) == Approx( 3.2034338050));
	CHECK(ps.overlap(6, 7) == Approx( 0.0000000000));

	CHECK(ps.overlap(7, 0) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 1) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 2) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 3) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 4) == Approx(-4.8380555e-01));
	CHECK(ps.overlap(7, 5) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 6) == Approx( 0.0000000000));
	CHECK(ps.overlap(7, 7) == Approx( 3.2034338050));

  }

  SECTION("PSP8 pseudopotential file"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "78_Pt_r.oncvpsp.psp8", sep, gcutoff, false);

    CHECK(ps.valence_charge() == 18.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 3.09_a);
		
    //values validated with Octopus
    CHECK(ps.valence_charge()*sep.long_range_potential(0.00000000E+00) == -2.29790754E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-04) == -2.29790753E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-02) == -2.29780949E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-02) == -2.29545879E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-01) == -2.28814066E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-01) == -2.07464113E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-00) == -1.60272255E+01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(4.99000000E+00) == -3.60721443E+00_a);

    CHECK(ps.short_range_potential().function()(0.00000000E+00) == 5.77774525E+00_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-02) == 5.77700949E+00_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-02) == 5.75937414E+00_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-01) == 5.70458232E+00_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-01) == 4.18518028E+00_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-00) == 1.52278621E+00_a);
    CHECK(ps.short_range_potential().function()(4.99000000E+00) == 8.23104510E-07_a);

    CHECK(ps.projector_radius() == 3.03_a);

		CHECK(not ps.has_electronic_density());

		CHECK(not ps.has_nlcc_density());
		
		CHECK(not ps.has_overlap());
  }

  SECTION("QSO/Qbox pseudopotential file"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "C_ONCV_PBE-1.2.xml", sep, gcutoff);

    CHECK(ps.valence_charge() == 4.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 3.06_a);
		
    //values validated with Octopus
    CHECK(ps.valence_charge()*sep.long_range_potential(0.00000000E+00) == -5.10646119E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-04) == -5.10646116E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-02) == -5.10624332E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-02) == -5.10101952E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-01) == -5.08475703E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-01) == -4.61031362E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(1.00000000E-00) == -3.56160567E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(5.00000000E-00) == -8.00000000E-01_a);
    CHECK(ps.valence_charge()*sep.long_range_potential(6.01000000E-00) == -6.65557404E-01_a);

    CHECK(ps.valence_charge()*sep.long_range_density(0.00000000E+00) == -1.04027973E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-02) == -1.04014658E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-02) == -1.03695616E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-01) == -1.02704901E+00_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-01) == -7.55398125E-01_a);
    CHECK(ps.valence_charge()*sep.long_range_density(1.00000000E-00) == -2.89236568E-01_a);
    CHECK(ps.valence_charge()*sep.long_range_density(5.00000000E-00) == -1.31742747E-14_a);
    CHECK(ps.valence_charge()*sep.long_range_density(6.01000000E-00) == -8.67155910E-21_a);
    
    CHECK(ps.short_range_potential().function()(0.00000000E+00) == -5.0051044619_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-02) == -5.0041488333_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-02) == -4.9721917357_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-01) == -4.8724398846_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-01) == -2.5917381777_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-00) == -0.4429701602_a);

    CHECK(ps.projector_radius() == 1.94_a);

		CHECK(not ps.has_electronic_density());

		CHECK(not ps.has_nlcc_density());
		CHECK(not ps.has_overlap());

  }
	
  SECTION("UPF nitrogen"){
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "N.upf", sep, gcutoff);

    CHECK(ps.valence_charge() == 5.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 3.25_a);

		CHECK(ps.num_projectors_l() == 4);
    CHECK(ps.num_projectors_lm() == 8);

		CHECK(ps.projector_l(0) == 0);
		CHECK(ps.projector_l(1) == 0);
		CHECK(ps.projector_l(2) == 1);
		CHECK(ps.projector_l(3) == 1);

		CHECK(ps.kb_coeff(0, 0) ==  7.4945088150_a);
		CHECK(ps.kb_coeff(1, 1) ==  0.6363049519_a);
		CHECK(ps.kb_coeff(2, 2) == -4.2939052122_a);
		CHECK(ps.kb_coeff(3, 3) == -1.0069878791_a);

		for(int ii = 0; ii < 4; ii++){
			for(int jj = 0; jj < 4; jj++){
				if(ii == jj) continue;
				CHECK(ps.kb_coeff(ii, jj) == 0.0_a);
			}
		}
		
    CHECK(ps.short_range_potential().function()(0.00000000E+00) == -2.0578140388_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-02) == -2.0580103342_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-02) == -2.0589810233_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-01) == -2.0608540875_a);
    CHECK(ps.short_range_potential().function()(5.00000000E-01) == -1.8015597502_a);
    CHECK(ps.short_range_potential().function()(1.00000000E-00) == -0.5354504615_a);
		CHECK(ps.short_range_potential().function()(2.00000000E-00) == -0.0032677043_a);
		CHECK(ps.short_range_potential().function()(3.00000000E-00) == -0.0000245461_a);

    CHECK(ps.projector_radius() == 2.0_a);
    
    CHECK(ps.projector(0).function()(0.00000000E+00) ==  9.648155811_a);
    CHECK(ps.projector(0).function()(1.00000000E-02) ==  9.6446013346_a);
    CHECK(ps.projector(0).function()(5.00000000E-02) ==  9.5250832071_a);
    CHECK(ps.projector(0).function()(1.00000000E-01) ==  9.1584198921_a);
    CHECK(ps.projector(0).function()(5.00000000E-01) ==  2.0281233377_a);
    CHECK(ps.projector(0).function()(1.00000000E-00) == -0.1538710364_a);
    CHECK(ps.projector(0).function()(1.20000000E+00) == -0.0049071428_a);

		CHECK(ps.projector(1).function()(0.00000000E+00) == -6.9511293171_a);
		CHECK(ps.projector(1).function()(1.00000000E-02) == -6.9436453361_a);
    CHECK(ps.projector(1).function()(5.00000000E-02) == -6.7408367692_a);
    CHECK(ps.projector(1).function()(1.00000000E-01) == -6.12731016_a);
    CHECK(ps.projector(1).function()(5.00000000E-01) ==  2.523437651_a);
    CHECK(ps.projector(1).function()(1.00000000E-00) ==  0.2295145138_a);
    CHECK(ps.projector(1).function()(1.20000000E+00) == -0.0516565314_a);
		
		CHECK(ps.projector(2).function()(0.00000000E+00) ==  0.0_a);
		CHECK(ps.projector(2).function()(1.00000000E-02) ==  0.2771729105_a);
    CHECK(ps.projector(2).function()(5.00000000E-02) ==  1.368551868_a);
    CHECK(ps.projector(2).function()(1.00000000E-01) ==  2.6309655738_a);
    CHECK(ps.projector(2).function()(5.00000000E-01) ==  3.1572749558_a);
    CHECK(ps.projector(2).function()(1.00000000E-00) == -0.3504668646_a);
    CHECK(ps.projector(2).function()(1.20000000E+00) == -0.0422974013_a);
		
		CHECK(ps.projector(3).function()(0.00000000E+00) ==  0.0_a);
		CHECK(ps.projector(3).function()(1.00000000E-02) == -0.2017723798_a);
    CHECK(ps.projector(3).function()(5.00000000E-02) == -0.9872054615_a);
    CHECK(ps.projector(3).function()(1.00000000E-01) == -1.842813857_a);
    CHECK(ps.projector(3).function()(5.00000000E-01) ==  0.9804124041_a);
    CHECK(ps.projector(3).function()(1.00000000E-00) ==  0.4937147551_a);
    CHECK(ps.projector(3).function()(1.20000000E+00) == -0.159129482_a);

		CHECK(ps.has_electronic_density());

		CHECK(ps.electronic_density_radius() == 5.9_a);
		CHECK(ps.electronic_density().function()(0.00000000E+00) == 0.0811765885_a);
    CHECK(ps.electronic_density().function()(1.00000000E-02) == 0.0815458437_a);
    CHECK(ps.electronic_density().function()(5.00000000E-02) == 0.0902259790_a);
    CHECK(ps.electronic_density().function()(1.00000000E-01) == 0.1164456375_a);
    CHECK(ps.electronic_density().function()(5.00000000E-01) == 0.4789764596_a);
    CHECK(ps.electronic_density().function()(1.00000000E-00) == 0.2768625568_a);
		CHECK(ps.electronic_density().function()(2.00000000E-00) == 0.0263067963_a);
		CHECK(ps.electronic_density().function()(3.00000000E-00) == 0.0027003340_a);

		CHECK(ps.has_nlcc_density());

		CHECK(ps.nlcc_density_radius() == 1.25_a);
		CHECK(ps.nlcc_density().function()(0.00000000E+00) == 1.9097357900_a);
    CHECK(ps.nlcc_density().function()(1.00000000E-02) == 1.9084160514_a);
    CHECK(ps.nlcc_density().function()(5.00000000E-02) == 1.8773513882_a);
    CHECK(ps.nlcc_density().function()(1.00000000E-01) == 1.7832982946_a);
    CHECK(ps.nlcc_density().function()(5.00000000E-01) == 0.3109042092_a);
    CHECK(ps.nlcc_density().function()(1.00000000E-00) == 0.0000094236_a);
		
  }

	
  SECTION("UPF nitrogen non-diagonal"){
		
    pseudo::pseudopotential ps(pseudopod::path::unit_tests_data() + "N_non_diagonal.upf.gz", sep, gcutoff);

    CHECK(ps.valence_charge() == 5.0_a);

    CHECK(sep.long_range_density_radius() == 5.0922713869_a);

    CHECK(ps.short_range_potential_radius() == 3.25_a);

		CHECK(ps.num_projectors_l() == 4);
    CHECK(ps.num_projectors_lm() == 8);

		CHECK(ps.projector_l(0) == 0);
		CHECK(ps.projector_l(1) == 0);
		CHECK(ps.projector_l(2) == 1);
		CHECK(ps.projector_l(3) == 1);

		CHECK(ps.kb_coeff(0, 0) ==  4.0654068834_a);
		CHECK(ps.kb_coeff(0, 1) == -3.4291019316_a);
		CHECK(ps.kb_coeff(0, 2) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(0, 3) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(1, 0) == -3.4291019316_a);
		CHECK(ps.kb_coeff(1, 1) ==  4.0654068834_a);
		CHECK(ps.kb_coeff(1, 2) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(1, 3) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(2, 0) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(2, 1) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(2, 2) == -2.6504465456_a);
		CHECK(ps.kb_coeff(2, 3) ==  1.6434586665_a);
		CHECK(ps.kb_coeff(3, 0) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(3, 1) ==  0.0000000000_a);
		CHECK(ps.kb_coeff(3, 2) ==  1.6434586665_a);
		CHECK(ps.kb_coeff(3, 3) == -2.6504465456_a);
		
    CHECK(ps.projector_radius() == 1.85_a);
		CHECK(ps.has_electronic_density());
		CHECK(ps.has_nlcc_density());

  }
}

#endif
