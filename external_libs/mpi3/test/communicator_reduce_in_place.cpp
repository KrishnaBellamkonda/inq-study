// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <cassert>
#include <functional>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	{
		int n = 1;
		world.reduce_in_place_n(&n, 1, std::plus<>{});
		if(world.rank() == 0) {assert(n == world.size());}
	}
	{
		int n = 1;
		world.all_reduce_in_place_n(&n, 1, std::plus<>{});
		assert(n == world.size());
	}
	// {
	// 	int n = 1;
	// 	world.all_reduce_n(&n, 1, std::plus<>{});
	// 	assert(n == world.size());
	// }
	// {
	// 	int n = 1;
	// 	world.all_reduce_n(&n, 1);
	// 	assert(n == world.size());
	// }
	{
		int const n = 1;
		auto const m = world.all_reduce_value(n);
		assert( n == 1 );  // cppcheck-suppress knownConditionTrueFalse
		assert(m == world.size());
	}
	{
		int const n = 1;
		auto const m = (world += n);
		assert( n == 1 );  // cppcheck-suppress knownConditionTrueFalse
		assert(m == world.size());
	}
//  {
//      int n = 1;
//      auto const m = (world + n);
//      assert( n == 1 );
//      assert(m == world.size());
//  }
}
