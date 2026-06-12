// Copyright 2020-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>
#include <mpi3/process.hpp>

#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner) used indrectly

#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

struct long_long {
	long long  value;                    // NOLINT(google-runtime-int,misc-non-private-member-variables-in-classes) testing type
	long_long& operator=(long long v) {  // NOLINT(google-runtime-int) testing type
		value = v;
		return *this;
	}
};

template<class Archive>
static void serialize(Archive& ar, long_long& l, unsigned /*version*/ = 0) {  // NOLINT(misc-use-anonymous-namespace) used by boost.serialization  // cppcheck-suppress [unusedFunction,unmatchedSuppression] ; false positive in cppcheck 2.11
	ar & l.value;
}

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	assert(world.size() > 1);
	constexpr long long size = 10000;  // 000;  // NOLINT(google-runtime-int) testing type
	switch(world.rank()) {
	case 0: {
		std::vector<long_long> v(static_cast<std::size_t>(size));
		std::iota(v.begin(), v.end(), 0.0);  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota
		//  assert(std::accumulate(v.begin(), v.end(), 0.) == size*(size-1)/2 );
		world[1] << v;
	} break;
	case 1: {
		std::vector<long_long> w;
		world[0] >> w;
		assert(w.size() == static_cast<std::size_t>(size));
		assert(w[45].value == 45);
		//  assert(std::accumulate(w.begin(), w.end(), 0.) == size*(size-1)/2 );
	} break;
	default: {
	}
	}
}
