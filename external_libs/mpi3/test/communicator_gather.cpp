// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <mpi3/process.hpp>
#include <mpi3/vector.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <utility>
#include <vector>

namespace mpi3 = boost::mpi3;

namespace {

void part1(mpi3::communicator& world) {
	std::vector<std::pair<double, int>> small(10, {0.0, world.rank()});
	std::vector<std::pair<double, int>> large(world.root() ? small.size() * static_cast<std::size_t>(world.size()) : 0, std::pair<double, int>(0.0, -10));

	auto it = world.gather_n(small.begin(), small.size(), large.begin(), 0);
	assert(it == large.end());

	if(world.rank() == 0) {
		assert(it != large.begin());
		assert((large[9] == std::pair<double, int>(0.0, 0)));
		assert((large[11] == std::pair<double, int>(0.0, 1)));
	} else {
		assert(it == large.begin());
	}
}

void part2(mpi3::communicator& world) {
	std::vector<std::pair<double, int>> small(10, {0., world.rank()});
	std::vector<std::pair<double, int>> large(world.root() ? small.size() * static_cast<std::size_t>(world.size()) : 0, std::pair<double, int>(0., -1));

	auto it = world.gather_n(small.begin(), small.size(), large.begin());
	assert(it == large.end());

	if(world.root()) {
		assert(it != large.begin());
		assert((large[9] == std::pair<double, int>(0., 0)));
		assert((large[11] == std::pair<double, int>(0., 1)));
	} else {
		assert(it == large.begin());
	}
}

void part3(mpi3::communicator& world) {
	std::list<double> small(10, world.rank());
	std::vector<double> large(world.root()?small.size()*static_cast<std::size_t>(world.size()):0, -1.0);

	world.gather(small.begin(), small.end(), large.begin(), 0);
	if(world.root()) {
		std::cout << "large: ";
		std::copy(large.begin(), large.end(), std::ostream_iterator<double>(std::cout, ", "));  // NOLINT(boost-use-ranges) for C++20, use std::ranges::copy
		std::cout << '\n';

		assert(large[ 1] == 0);
		assert(large[11] == 1);
		assert(large[21] == 2);
	}
}

// void part4(mpi3::communicator& world) {
//  auto val = std::string{"5.1 opa"};

//  using T = decltype(val);

//  std::vector<T> small(10, val);
//  std::vector<T> large(world.root() ? small.size() * static_cast<std::size_t>(world.size()) : 0);

//  world.gather(small.begin(), small.end(), large.begin(), 0);

//  if(world.rank() == 0) {
//      assert(all_of(large.begin(), large.end(), [val](auto& e) { return val == e; }));
//  }
// }

#ifndef EXAMPI
void part5(mpi3::communicator& world) {
	auto Lval  = std::to_string(world.rank() + 1000);
	auto vals0 = (world[0] |= Lval);
	if(world.rank() == 0) {
		assert(vals0.size() - static_cast<std::size_t>(world.size()) == 0);
		assert(vals0[2] == "1002");
	} else {
		assert(vals0.empty());
	}
}
#endif

}  // end namespace 

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	mpi3::vector<double> const v;

	assert( world.size() > 2);

	part1(world);
	part2(world);
	part3(world);
//  part4(world);
#ifndef EXAMPI
	part5(world);
#endif
}
