// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>  // NOLINT(misc-include-cleaner) for using literal, e.g. 5s
#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	assert(world.size() > 1);

	{
		std::vector<double> small(10, 5.);
		iota(begin(small), end(small), world.rank());  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota
		std::vector<double> large;
		if(world.rank() == 0) {
			large.resize(small.size() * static_cast<std::size_t>(world.size()), -1.0);
		}
		{
			auto req = world.igather_n(small.begin(), small.size(), large.begin(), 0);
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(2s);  // NOLINT(misc-include-cleaner) bug in clang-tidy 18.1
											  //  req.wait();
		}
		if(world.rank() == 0) {
			assert(equal(begin(large), begin(large) + static_cast<std::ptrdiff_t>(small.size()), begin(small)));
		}
	}
	{
		std::vector<double> small(10, 5.);
		std::vector<double> large(small.size() * static_cast<std::size_t>(world.size()), -1.0);
		{
			auto req = world.iall_gather_n(small.begin(), small.size(), large.begin());
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(5s);  // NOLINT(misc-include-cleaner) bug in clang-tidy 18.1.3
		}
		assert(std::all_of(large.begin(), large.end(), [](auto const& e) { return 5.0 == e; }));  // NOLINT(boost-use-ranges) for C++20, use std::ranges::all_of
	}
}
