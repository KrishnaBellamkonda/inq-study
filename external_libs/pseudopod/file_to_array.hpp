/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDO_FILE_TO_ARRAY_HPP
#define PSEUDO_FILE_TO_ARRAY_HPP

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <vector>
#include <string>
#include <rapidxml/rapidxml.hpp>
#include "formats/base.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>

#include <zlib.h>

#include <fstream>
#include <iostream>

namespace pseudo {

static auto file_to_array(const std::string & filename){
  
	std::string extension = filename.substr(filename.find_last_of(".") + 1);
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	std::vector<char> buffer;
	
  if(extension == "gz"){
		gzFile file = gzopen(filename.c_str(), "rb");
		if(!file) throw std::runtime_error{"pseudopod error: cannot open compressed file '" + filename + "'."};
		
		char charbuff[1024];
    int num_read = 0;
		
    while ((num_read = gzread(file, charbuff, sizeof(charbuff))) > 0) {
			for(int ii = 0; ii < num_read; ii++) buffer.emplace_back(charbuff[ii]);
		}

		gzclose(file);
		
  } else {

		std::ifstream file(filename);
		if(!file) throw std::runtime_error{"pseudopod error: cannot open file '" + filename + "'."};
		buffer = std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  }

	buffer.push_back('\0');
	return buffer;

}

}

#endif

#ifdef INQ__FILE_TO_ARRAY_UNIT_TEST
#undef INQ__FILE_TO_ARRAY_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;

  SECTION("Uncompressed"){

    auto arr = pseudo::file_to_array(pseudopod::path::unit_tests_data() + "hola.txt");

    CHECK(arr.size() == 6);
    
    CHECK(arr[0] == 'h');
    CHECK(arr[1] == 'o');    
    CHECK(arr[2] == 'l');
    CHECK(arr[3] == 'a');
    CHECK(arr[4] == '\n');
    CHECK(arr[5] == '\0');    
  }
	
  SECTION("Missing uncompressed"){
		CHECK_THROWS(pseudo::file_to_array(pseudopod::path::unit_tests_data() + "chao.txt"));
	}
  
  SECTION("Compressed"){
    
    auto arr = pseudo::file_to_array(pseudopod::path::unit_tests_data() + "hola.txt.gz");

    CHECK(arr.size() == 6);
    
    CHECK(arr[0] == 'h');
    CHECK(arr[1] == 'o');    
    CHECK(arr[2] == 'l');
    CHECK(arr[3] == 'a');
    CHECK(arr[4] == '\n');
    CHECK(arr[5] == '\0');    
  }

  SECTION("Missing compressed"){
		CHECK_THROWS(pseudo::file_to_array(pseudopod::path::unit_tests_data() + "chao.txt.gz"));		
	}
  
}
#endif
