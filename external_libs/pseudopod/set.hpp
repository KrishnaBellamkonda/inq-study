/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_SET_HPP
#define PSEUDO_SET_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <string>
#include <map>
#include <fstream>
#include <stdexcept>

#include "element.hpp"

#include <iostream>
#include <dirent.h>
#include "detect_format.hpp"
#include "formats/psml.hpp"
#include "formats/qso.hpp"
#include "formats/upf1.hpp"
#include "formats/upf2.hpp"
#include "formats/psp8.hpp"
#include "set_id.hpp"

namespace pseudo {

class set{

private:

	struct element_values {
		std::string file_path_;
		int lmax_;
		int llocal_;
		double spacing_;
		double radius_;
	};

    
	typedef std::map<std::string, element_values> element_map;

	std::string path_;
	element_map map_;
	bool automatic_;
    
public:

	set(set_id id):
		set(id.path()){
	}
	
	set(std::string dirname, const bool automatic = true):
		path_(dirname),
		automatic_(automatic){

		dirname = pseudopod::path::share() + "/" + dirname;
			
		DIR * dir = opendir(dirname.c_str());

		if(dir == NULL)	throw std::runtime_error("pseudopod error: cannot open pseudopotential set directory '" + dirname + "'.");
		
		struct dirent *ent;
		while ((ent = readdir(dir)) != NULL) {
			const std::string filename(ent->d_name);
			const std::string fullname = dirname + "/" + filename;

			if(filename == "." || filename == "..") continue;

			pseudo::format format = detect_format(fullname);
	
			if(format == pseudo::format::FILE_NOT_FOUND || format == pseudo::format::UNKNOWN) continue;

			std::string symbol;
   
			switch(format){
			case pseudo::format::QSO:
				symbol = pseudo::qso(fullname).symbol();				
				break;
			case pseudo::format::UPF1:
				symbol = pseudo::upf1(fullname, /*uniform_grid = */ true).symbol();
				break;
			case pseudo::format::UPF2:
				symbol = pseudo::upf2(fullname, /*uniform_grid = */ true).symbol();
				break;
			case pseudo::format::PSML:
				symbol = pseudo::psml(fullname, /*uniform_grid = */ true).symbol();
				break;
			case pseudo::format::PSP8:
				symbol = pseudo::psp8(fullname).symbol();
				break;
			default:
				//get the symbol from the name
				for(int ii = 0; ii < 3; ii++){
					char cc = filename[ii];
					bool is_letter = (cc >= 'a' && cc <= 'z') || (cc >= 'A' && cc <= 'Z');
					if(!is_letter) break;
					symbol.push_back(cc);
				}
			}

			if(map_.find(symbol) != map_.end()){
				std::string msg = "\n\nERROR: duplicated pseudopotential for '" + symbol + "':\n";
				msg += "  directory      = '" + dirname + "'\n";
				msg += "  original file  = '" + map_[symbol].file_path_ + "'\n";
				msg += "  duplicate file = '" + fullname + "'\n";
				throw std::runtime_error(msg);
			}
			
			element_values vals;

			vals.file_path_ = fullname;
			vals.lmax_ = INVALID_L;
			vals.llocal_ = INVALID_L;
			vals.spacing_ = -1.0;
			vals.radius_ = -1.0;

			map_[symbol] = vals;

		}

		std::ifstream defaults_file((dirname + "/set_defaults").c_str() );

		if(defaults_file){
			std::string line;

			//first line are comments
			getline(defaults_file, line);

			while(true){
				std::string symbol;
				defaults_file >> symbol;
				if(defaults_file.eof()) break;

				if(has(symbol)){
					int z;
					std::string fname;
	    
					defaults_file >> fname;
					defaults_file >> z;
					defaults_file >> map_[symbol].lmax_;
					defaults_file >> map_[symbol].llocal_;
					defaults_file >> map_[symbol].spacing_;
					defaults_file >> map_[symbol].radius_;
				}

				getline(defaults_file, line);
			}
	
			defaults_file.close();
		}

		closedir(dir);
		if(map_.empty()) throw std::runtime_error("pseudopod error: the directory '" + dirname + "' does not contain any pseudopotential files.");
	}
    
	bool has(const element & el) const {
		return map_.find(el.symbol()) != map_.end();
	}
    
	const std::string & file_path(const element & el) const {
		return map_.at(el.symbol()).file_path_;
	}
    
	int lmax(const element & el) const {
		return map_.at(el.symbol()).lmax_;
	}
    
	int llocal(const element & el) const {
		return map_.at(el.symbol()).llocal_;
	}
    
	double spacing(const element & el, double etol) const {
		std::ifstream file((map_.at(el.symbol()).file_path_ + ".spacing").c_str());
		if(automatic_ && file) {
			std::vector<double> spacing;
			std::vector<double> energy;

			while(true){
				double h, e;
				std::string line;
				file >> h >> e;
				getline(file, line);
				if(file.eof()) break;
				spacing.push_back(h);
				energy.push_back(e);
			}

			double eref = energy[energy.size() - 1];
			energy[energy.size() - 1] = 0.0;
			for(int ii = energy.size() - 2; ii >= 0; ii--){
				energy[ii] = std::max(energy[ii + 1], fabs(energy[ii] - eref));
			}

			{
				//make the curve a bit smoother by taking a running average
				std::vector<double> e2(energy);
				energy[1] = (e2[0] + e2[1] + e2[2])/3.0;
				for(unsigned ii = 2; ii < energy.size() - 2; ii++){
					energy[ii] = (e2[ii - 2] + e2[ii - 1] + e2[ii] + e2[ii + 1] + e2[ii + 2])/5.0;
				}
				energy[energy.size() - 2] = (e2[energy.size() - 3] + e2[energy.size() - 2] + e2[energy.size() - 1])/3.0;
			}

			for(unsigned ii = 0; ii < energy.size(); ii++){
				if(energy[ii] < etol) return spacing[ii];
			}
		
		}
			
		return map_.at(el.symbol()).spacing_;
	}
    
	double radius(const element & el) const {
		return map_.at(el.symbol()).radius_;
	}

	auto size() const {
		return map_.size();
	}

	//Iterator interface

	class iterator {

	private:
		element_map::iterator map_it_;

	public:
		iterator(const element_map::iterator & map_it):map_it_(map_it){
		}

		iterator & operator++(){
			++map_it_;
			return *this;
		}

		friend bool operator!=(const iterator & a, const iterator & b){
			return a.map_it_ != b.map_it_;
		}
      
		element operator*(){
			return element(map_it_->first);
		}

	};
    
	iterator begin(){ return iterator(map_.begin()); }
	iterator end(){ return iterator(map_.end()); }

	auto & path() const {
		return path_;
	}
	
};

}

#endif

#ifdef INQ__SET_UNIT_TEST
#undef INQ__SET_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace pseudo;
	using namespace Catch::literals;
	using Catch::Approx;

	CHECK_THROWS(set("non_existing_directory"));
	CHECK_THROWS(set("./"));

	{
		auto pset = set(set_id::pseudodojo_pbe());
		CHECK(pset.size() == 72);
	}

	{
		auto pset = set(set_id::pseudodojo_pbe_stringent());
		CHECK(pset.size() == 72);
	}

	{
		auto pset = set(set_id::pseudodojo_lda());
		CHECK(pset.size() == 70);
	}

	{
		auto pset = set(set_id::pseudodojo_lda_stringent());
		CHECK(pset.size() == 70);
	}


	{
		auto pset = set(set_id::pseudodojo_pbesol());
		CHECK(pset.size() == 72);
	}

	{
		auto pset = set(set_id::pseudodojo_pbesol_stringent());
		CHECK(pset.size() == 72);
	}

	{
		auto pset = set(set_id::sg15());
		CHECK(pset.size() == 69);
	}

	{
		auto pset = set(set_id::ccecp());
		CHECK(pset.size() == 47);
	}
	
}
#endif
