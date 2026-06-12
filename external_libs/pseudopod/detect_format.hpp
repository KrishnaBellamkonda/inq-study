/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_DETECT_FORMAT_HPP
#define PSEUDO_DETECT_FORMAT_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <vector>
#include <string>
#include <rapidxml/rapidxml.hpp>
#include "file_to_array.hpp"
#include "formats/base.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

#include <fstream>

namespace pseudo {

static pseudo::format detect_format(const std::string & filename){

	// check that the file is not a directory
	struct stat file_stat;
	if(stat(filename.c_str(), &file_stat) == -1) return pseudo::format::FILE_NOT_FOUND;
	if(S_ISDIR(file_stat.st_mode)) return pseudo::format::FILE_NOT_FOUND;

	auto buffer = file_to_array(filename);

	{
		rapidxml::xml_document<> doc;
		bool is_xml = true;
		try {
			doc.parse<0>(&buffer[0]);
		} catch(rapidxml::parse_error & /*xml_error*/){
			is_xml = false;
		}
      
		if(is_xml){
			if(doc.first_node("fpmd:species")) return pseudo::format::QSO;
			if(doc.first_node("PP_INFO")) return pseudo::format::UPF1;
			if(doc.first_node("UPF")) return pseudo::format::UPF2;
			if(doc.first_node("psml")) return pseudo::format::PSML;
			if(doc.first_node("paw_dataset")) return pseudo::format::PAWXML;
		}
	}
	
	auto extension = filename.substr(filename.find_last_of(".") + 1);
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	if(extension == "gz") {
		auto woext = filename.substr(0, filename.find_last_of("."));
		extension = woext.substr(woext.find_last_of(".") + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
	}
	
	if(extension == "psp8") return pseudo::format::PSP8;
	if(extension == "drh")  return pseudo::format::PSP8;
	if(extension == "psf")  return pseudo::format::PSF;
	if(extension == "cpi")  return pseudo::format::CPI;
	if(extension == "fhi")  return pseudo::format::FHI;
	if(extension == "hgh")  return pseudo::format::HGH;
    
	return pseudo::format::UNKNOWN;
}

}

#endif

#ifdef INQ__DETECT_FORMAT_UNIT_TEST
#undef INQ__DETECT_FORMAT_UNIT_TEST

#include <catch2/catch_all.hpp>

#include "path.hpp"

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;

	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "W_ONCV_PBE-1.0.upf") == pseudo::format::UPF2);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "F.UPF") == pseudo::format::UPF1);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "I_HSCV_LDA-1.0.xml") == pseudo::format::QSO);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "78_Pt_r.oncvpsp.psp8") == pseudo::format::PSP8);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "78_Pt_r.oncvpsp.psp8.gz") == pseudo::format::PSP8);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "C_ONCV_PBE-1.2.xml") == pseudo::format::QSO);
	CHECK(pseudo::detect_format(pseudopod::path::unit_tests_data() + "Ca.LDA_PW-JTH.xml") == pseudo::format::PAWXML);

}
#endif
