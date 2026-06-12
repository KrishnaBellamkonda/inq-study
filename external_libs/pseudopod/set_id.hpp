/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_SET_ID_HPP
#define PSEUDO_SET_ID_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <string>
#include <iostream>

namespace pseudo {

class set_id {

	enum class id_enum {
		PSEUDODOJO_PBE               =  0,
		PSEUDODOJO_PBE_STRINGENT     =  1,
		PSEUDODOJO_LDA               =  2,
		PSEUDODOJO_LDA_STRINGENT     =  3,
		PSEUDODOJO_PBESOL            =  4,
		PSEUDODOJO_PBESOL_STRINGENT  =  5,
		PSEUDODOJO_REL_PBE           =  6,
		SG15                         =  7,
		SG15_REL                     =  8,
		CCECP                        =  9
	};

	constexpr static int NUM_IDS = 10;
	
	id_enum id_;

	set_id(id_enum arg_id):
		id_(arg_id){
	}
	
public:

	auto operator==(set_id const & other) const {
		return id_ == other.id_;
	}
	
	set_id():
		id_(id_enum::PSEUDODOJO_PBE){
	}

	set_id(std::string const & set_name) {
		if(set_name == "pseudodojo_pbe"){
			id_ = set_id::id_enum::PSEUDODOJO_PBE;
		} else if(set_name == "pseudodojo_pbe_stringent"){
			id_ = set_id::id_enum::PSEUDODOJO_PBE_STRINGENT;
		} else if(set_name == "pseudodojo_lda"){
			id_ = set_id::id_enum::PSEUDODOJO_LDA;
		} else if(set_name == "pseudodojo_lda_stringent"){
			id_ = set_id::id_enum::PSEUDODOJO_LDA_STRINGENT;
		} else if(set_name == "pseudodojo_pbesol"){
			id_ = set_id::id_enum::PSEUDODOJO_PBESOL;
		} else if(set_name == "pseudodojo_pbesol_stringent"){
			id_ = set_id::id_enum::PSEUDODOJO_PBESOL_STRINGENT;
		} else if(set_name == "pseudodojo_rel_pbe"){
			id_ = set_id::id_enum::PSEUDODOJO_REL_PBE;
		} else if(set_name == "sg15"){
			id_ = set_id::id_enum::SG15;
		} else if(set_name == "sg15_rel"){
			id_ = set_id::id_enum::SG15_REL;
		} else if(set_name == "ccecp"){
			id_ = set_id::id_enum::CCECP;
		} else {
			throw std::runtime_error("Pseudopod error: Invalid pseudopotential set identification '" + set_name + "'.");
		}
	}
	
	static auto pseudodojo_pbe(){
		return set_id{id_enum::PSEUDODOJO_PBE};
	}

	static auto pseudodojo_pbe_stringent(){
		return set_id{id_enum::PSEUDODOJO_PBE_STRINGENT};
	}
	
	static auto pseudodojo_lda(){
		return set_id{id_enum::PSEUDODOJO_LDA};
	}

	static auto pseudodojo_lda_stringent(){
		return set_id{id_enum::PSEUDODOJO_LDA_STRINGENT};
	}

	static auto pseudodojo_pbesol(){
		return set_id{id_enum::PSEUDODOJO_PBESOL};
	}

	static auto pseudodojo_pbesol_stringent(){
		return set_id{id_enum::PSEUDODOJO_PBESOL_STRINGENT};
	}

	static auto pseudodojo_rel_pbe(){
		return set_id{id_enum::PSEUDODOJO_REL_PBE};
	}
	
	static auto sg15(){
		return set_id{id_enum::SG15};
	}

	static auto sg15_rel(){
		return set_id{id_enum::SG15_REL};
	}

	static auto ccecp(){
		return set_id{id_enum::CCECP};
	}

	template<class OStream>
	friend OStream & operator<<(OStream & out, set_id const & self){
		
		if(self.id_ == set_id::id_enum::PSEUDODOJO_PBE){
			out << "pseudodojo_pbe";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_PBE_STRINGENT){
			out << "pseudodojo_pbe_stringent";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_LDA){
			out << "pseudodojo_lda";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_LDA_STRINGENT){
			out << "pseudodojo_lda_stringent";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_PBESOL){
			out << "pseudodojo_pbesol";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_PBESOL_STRINGENT){
			out << "pseudodojo_pbesol_stringent";
		} else if(self.id_ == set_id::id_enum::PSEUDODOJO_REL_PBE){
			out << "pseudodojo_rel_pbe";
		} else if(self.id_ == set_id::id_enum::SG15){
			out << "sg15";
		} else if(self.id_ == set_id::id_enum::SG15_REL){
			out << "sg15_rel";
		} else if(self.id_ == set_id::id_enum::CCECP){
			out << "ccecp";
		}
 		return out;
	}

	template<class IStream>
	friend IStream & operator>>(IStream & in, set_id & self){
		std::string readval;
		in >> readval;
		self = set_id(readval);
		return in;
	}
		
	auto path() const {
    using namespace std::string_literals;

		switch(id_) {
		case id_enum::PSEUDODOJO_PBE:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-05_pbe_standard_upf/"s;
		case id_enum::PSEUDODOJO_PBE_STRINGENT:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-04_pbe_stringent_upf/"s;
		case id_enum::PSEUDODOJO_LDA:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-04_pw_standard_upf/"s;
		case id_enum::PSEUDODOJO_LDA_STRINGENT:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-04_pw_stringent_upf/"s;
		case id_enum::PSEUDODOJO_PBESOL:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-04_pbesol_standard_upf/"s;
		case id_enum::PSEUDODOJO_PBESOL_STRINGENT:
			return "pseudopotentials/pseudo-dojo.org/nc-sr-04_pbesol_stringent_upf/"s;
		case id_enum::PSEUDODOJO_REL_PBE:
			return "pseudopotentials/pseudo-dojo.org/nc-fr-04_pbe_standard/"s;
		case id_enum::SG15:
			return "pseudopotentials/quantum-simulation.org/sg15/"s;
		case id_enum::SG15_REL:
			return "pseudopotentials/quantum-simulation.org/sg15_fr/"s;
		case id_enum::CCECP:
			return "pseudopotentials/pseudopotentiallibrary.org/ccecp/"s;
		}
		throw std::runtime_error("Pseudopod error: Invalid pseudopotential set id");
		return ""s;
	}

	static auto list() {
		std::vector<set_id> id_list;
		for(int ii = 0; ii < NUM_IDS; ii++) id_list.push_back(set_id{static_cast<id_enum>(ii)});
		return id_list;
	}
	
};

}

#endif

#ifdef INQ__SET_ID_UNIT_TEST
#undef INQ__SET_ID_UNIT_TEST

#include <catch2/catch_all.hpp>

auto check_io(pseudo::set_id const & id, std::string const & id_string) {
	std::stringstream ss;
	std::string str;
	ss << id << ' ' << id;
	ss >> str;
	CHECK(str == id_string);
	pseudo::set_id read_id;
	ss >> read_id;
	CHECK(id == read_id);
}

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace pseudo;
	using namespace Catch::literals;
	using Catch::Approx;

	SECTION("I/O"){
		check_io(set_id::pseudodojo_pbe(), "pseudodojo_pbe");
		check_io(set_id::pseudodojo_pbe_stringent(), "pseudodojo_pbe_stringent");
		check_io(set_id::pseudodojo_lda(), "pseudodojo_lda");
		check_io(set_id::pseudodojo_lda_stringent(), "pseudodojo_lda_stringent");
		check_io(set_id::pseudodojo_pbesol(), "pseudodojo_pbesol");
		check_io(set_id::pseudodojo_pbesol_stringent(), "pseudodojo_pbesol_stringent");
		check_io(set_id::pseudodojo_rel_pbe(), "pseudodojo_rel_pbe");
		check_io(set_id::sg15(), "sg15");
		check_io(set_id::sg15_rel(), "sg15_rel");		
		check_io(set_id::ccecp(), "ccecp");
	}

	SECTION("list") {
		auto list = set_id::list();
		CHECK(list.size() == 10);
	}

}
#endif
