/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__UTILS__VEC_FIELD
#define INQ__UTILS__VEC_FIELD

// Copyright (C) 2019-2024 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

namespace inq {
namespace utils {

	// convert std::vector objects to field or field_set objects


	template<typename ElementType, class BasisType>
	auto vec_to_field(std::vector<ElementType> const & vec, BasisType const & basis){
		basis::field<BasisType, ElementType> vec_field(basis);
		assert((long) vec.size() == vec_field.basis().size());
		assert(vec_field.basis().comm().size() == 1);  // serial case for now
		for(int i = 0; i < vec_field.basis().size(); i++) {
			vec_field.linear()[i] = vec[i];
		}
		return vec_field;
	}

	auto field_to_vec(basis::field<basis::real_space, double> const & vec_rs){
		std::vector<double> vec;
		vec.resize(vec_rs.basis().size());
		assert(vec_rs.basis().comm().size() == 1);  // serial case for now
		for(int i = 0; i < vec_rs.basis().size(); i++) {
			vec[i] = vec_rs.linear()[i];
		}
		return vec;
	}

	template<typename ElementType, class BasisType>
	auto vec_to_field_set(std::vector<ElementType> const & vec, BasisType const & basis, const int num_vectors){
		assert(num_vectors == 1); // nspin=1
		basis::field_set<BasisType, ElementType> vec_field_set(basis,num_vectors);
		assert((long long) vec.size() == vec_field_set.basis().size());
		assert(vec_field_set.basis().comm().size() == 1);  // serial case for now
		for(int i = 0; i < vec_field_set.basis().size(); i++) {
			for (int ivec = 0; ivec < num_vectors; ivec++){
				vec_field_set.matrix()[i][ivec] = vec[i];
			}
		}
		return vec_field_set;
	}

	template<typename ElementType, class BasisType>
	auto field_set_to_vec(basis::field_set<BasisType, ElementType> const & vec_field_set, const int num_vectors){
		std::vector<ElementType> vec;
		assert(num_vectors == 1); // nspin=1
		assert(vec_field_set.basis().comm().size() == 1);  // serial case for now
		vec.resize(vec_field_set.basis().size());
		for(int i = 0; i < vec_field_set.basis().size(); i++) {
			for (int ivec = 0; ivec < num_vectors; ivec++){
				vec[i] = vec_field_set.matrix()[i][ivec];
			}
		}
		return vec;
	}

}
}
#endif

#ifdef INQ_UTILS_VEC_FIELD_UNIT_TEST
#undef INQ_UTILS_VEC_FIELD_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <math/complex.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

}
#endif

