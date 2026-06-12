/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDOPOD__PATH
#define PSEUDOPOD__PATH

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <pseudopod_config.h>

#include <string>
#include <cstdlib>

namespace pseudopod {
namespace path {

std::string share(){
	auto env = std::getenv("PSEUDOPOD_SHARE_PATH");
	if(env == NULL) return PSEUDOPOD_PREFIX_DIR + std::string("/share/pseudopod/");
	return env + std::string("/");
}

std::string unit_tests_data(){
	return share() + std::string("unit_tests_data/");
}

}
}

#endif

#ifdef INQ__PATH_UNIT_TEST
#undef INQ__PATH_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;

	auto env = std::getenv("PSEUDOPOD_SHARE_PATH");
	std::string original_var;
	if(env != NULL) original_var = env;
	REQUIRE(unsetenv("PSEUDOPOD_SHARE_PATH") == 0);
	
	SECTION("Share path"){
    CHECK(pseudopod::path::share() == PSEUDOPOD_PREFIX_DIR + std::string("/share/pseudopod/"));
  }

	SECTION("Unit tests data path"){
    CHECK(pseudopod::path::unit_tests_data() == PSEUDOPOD_PREFIX_DIR + std::string("/share/pseudopod/unit_tests_data/"));
  }

	SECTION("Path from environment variable"){
		REQUIRE(setenv("PSEUDOPOD_SHARE_PATH", "/basura", 1) == 0);
    CHECK(pseudopod::path::share() == std::string("/basura/"));
		REQUIRE(unsetenv("PSEUDOPOD_SHARE_PATH") == 0);
  }

	if(not original_var.empty()) {
		REQUIRE(setenv("PSEUDOPOD_SHARE_PATH", original_var.c_str(), 1) == 0);
	}

}

#endif
