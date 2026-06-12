// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>
#include <mpi3/process.hpp>

#include <cassert>
#include <iterator>
#include <numeric>
#include <vector>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	{
		using T = int;
		std::vector<T> v(10);
		if(world.is_root()) {iota(begin(v), end(v), 0);}  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota

		world.broadcast(begin(v), end(v));
		assert( v[9] == T(9) );
	}
	{
		using T = double;
		std::vector<T> v(10);
		if(world.is_root()) {
			iota(begin(v), end(v), 0);  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota
		}
		world.broadcast(begin(v), end(v));

		assert( v[9] == T(9) );
	}

	{
		using T = double;
		std::vector<T> v;
		if(world.is_root()) { v.resize(10); iota(begin(v), end(v), 0); }  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota

		auto size = v.size();
		world.broadcast_n(&size, 1);

		v.resize(size);
		assert( v.size() == 10UL );

		world.broadcast_n(v.data(), v.size());

		assert( v[9] == T(9) );
	}
	{
		using T = double;
		std::vector<T> v;
		if(world.is_root()) {v.resize(10); iota(begin(v), end(v), 0);}  // NOLINT(boost-use-ranges) for C++20, use std::ranges::iota

		world[0] & v;

		assert( v.size() == 10UL );
		assert( v[9] == T(9) );
	}
}
