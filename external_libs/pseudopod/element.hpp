/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_ELEMENT_HPP
#define PSEUDO_ELEMENT_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <string>
#include <cassert>
#include <cctype>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <map>
#include <cstdlib>

#include "path.hpp"

namespace pseudo {
  
class element {

private:

	struct properties;
	typedef std::map<std::string, properties> map_type;
    
public:
    
	element(const std::string & symbol = "none"):
		symbol_(trim(symbol)) {
		
		if(symbol_.size() > 0) symbol_[0] = std::toupper(symbol_[0]);
		for(unsigned ii = 1; ii < symbol_.size(); ii++) symbol_[ii] = std::tolower(symbol_[ii]);

		valid(); //make sure the map is initialized
	}

	element(int atomic_number){

		//special case: avoid ambiguity between isotopes
		if(atomic_number == 1){
			symbol_ = 'H';
			return;
		}
      
		for(map_type::iterator it = map().begin(); it != map().end(); ++it){
			if(it->second.z_ == atomic_number) {
				symbol_ = trim(it->first);
				break;
			}
		}
	}
    
	bool valid() const {
		return map().find(symbol_) != map().end();
	}

	double charge() const {
		return -1.0*atomic_number();
	}

	const std::string & symbol() const{
		return symbol_;
	}
    
	int atomic_number() const {
		return map().at(symbol_).z_;
	}
    
	double mass() const{
		return map().at(symbol_).mass_;
	}

	double vdw_radius() const{
		return map().at(symbol_).vdw_radius_;
	}

	bool operator==(char const * el_symbol) const {
		return symbol() == el_symbol;
	}
	
	bool operator==(std::string const & el_symbol) const {
		return symbol() == el_symbol;
	}

	template <class element_type>
	bool operator==(const element_type & el) const {
		return symbol() == el.symbol();
	}

	struct hash {
		auto operator()(const element & el){
			std::hash<std::string> str_hash;
			return str_hash(el.symbol());
		}
	};
    
private:

	struct properties {
		int z_;
		double mass_;
		double vdw_radius_;
	};
    
	static map_type & map(){
      
		static map_type map;
		if(map.empty()){

			std::string filename = pseudopod::path::share() + "/pseudopotentials/elements.dat";
	
			std::ifstream file(filename.c_str());

			if(!file){
				std::cerr << "Internal error: cannot open file '" << filename << "'." << std::endl;
				exit(EXIT_FAILURE);
			}
	
			while(true){
				std::string symbol;
	  
				file >> symbol;

				if(file.eof()) break;
	  
				if(symbol[0] == '#'){
					getline(file, symbol);
					continue;
				}
	  
				properties prop;
	  
				file >> prop.z_ >> prop.mass_ >> prop.vdw_radius_;

				if(file.eof()) break;
	  
				map[symbol] = prop;
	  
			}
		}

		return map;
	}
    
	static std::string & ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r "){
		str.erase(0, str.find_first_not_of(chars));
		return str;
	}
 
	static std::string & rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r "){
		str.erase(str.find_last_not_of(chars) + 1);
		return str;
	}

public:
    
	static std::string trim(std::string str, const std::string& chars = "\t\n\v\f\r "){
		return ltrim(rtrim(str, chars), chars);
	}
    
private:
    
	std::string symbol_;
  
};

}


#ifdef INQ__ELEMENT_UNIT_TEST
#undef INQ__ELEMENT_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;
  using Catch::Matchers::WithinULP;

  SECTION("Define element by atomic number"){
    pseudo::element el(27);

    CHECK(el == pseudo::element("Co"));
    CHECK(el.valid());
    CHECK(el.atomic_number() == 27);
    CHECK(el.symbol() == "Co");
    CHECK_THAT(el.charge(), WithinULP(-27.0, 1));
    CHECK_THAT(el.mass(), WithinULP(58.9331944, 1));
    CHECK_THAT(el.vdw_radius(), WithinULP(4.535342, 1));
  }
  
  SECTION("Define element by symbol"){
    pseudo::element el("I");

    CHECK(el == pseudo::element(53));
    CHECK(el.valid());
    CHECK(el.atomic_number() == 53);
    CHECK(el.symbol() == "I");
    CHECK_THAT(el.charge(), WithinULP(-53.0, 1));
    CHECK_THAT(el.mass(), WithinULP(126.904473, 1));
    CHECK_THAT(el.vdw_radius(), WithinULP(3.855041, 1));
  }
  
  SECTION("Invalid element by atomic number"){
    pseudo::element el(200);

    CHECK(!el.valid());
  }

  SECTION("Invalid element by symbol"){
    pseudo::element el("X");

    CHECK(!el.valid());
  }
  
}
#endif

#endif
