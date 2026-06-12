// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <vector>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	assert( world.size() > 2 );
	{
		using T = std::tuple<double, double>;
		std::vector<T> v_local(10, T{world.rank(), world.rank()});
		std::vector<T> v(v_local.size() * static_cast<std::size_t>(world.size()));
		auto           end = world.all_gather_n(v_local.begin(), v_local.size(), v.begin());
		assert(end == v.end());
		assert(( v[ 0] == T{0.0, 0.0} ));
		assert(( v[ 9] == T{0.0, 0.0} ));
		assert(( v[10] == T{1.0, 1.0} ));
		assert(( v[19] == T{1.0, 1.0} ));
		assert(( v[20] == T{2.0, 2.0} ));
		assert(( v[29] == T{2.0, 2.0} ));
	}
	{
		using T = std::tuple<double, double>;
		std::vector<T> v_local(static_cast<std::size_t>(world.rank() + 5), T{world.rank(), world.rank()});
		std::vector<T> v(1000, T{-99.0, -99.0});
		auto           d_last = world.all_gatherv_n(begin(v_local), v_local.size(), begin(v));

		int const predict_size = (world.size() * (world.size() - 1) / 2) + (5 * world.size());

		assert( std::distance(begin(v), d_last) == predict_size );

		assert(( v[ 0] == T{0.0, 0.0} ));
		assert(( v[ 4] == T{0.0, 0.0} ));
		assert(( v[ 5] == T{1.0, 1.0} ));
		assert(( v[10] == T{1.0, 1.0} ));
		assert(( v[11] == T{2.0, 2.0} ));
		assert(( v[17] == T{2.0, 2.0} ));
	}
	{
		using T = std::tuple<double, double>;
		std::vector<T> v_local(static_cast<std::size_t>(world.rank() + 5), T{world.rank(), world.rank()});
		std::vector<T> v(1000, T{-99.0, -99.0});
		auto           d_last = world.all_gatherv(begin(v_local), end(v_local), begin(v));
		assert( d_last < end(v) );
	}
	{
		// initialize data
		using T = double;
		assert( world.size() == 3 );
		std::vector<T> v_loc;
		switch(world.rank()) {
		case 0: v_loc = {0.0, 0.0, 0.0}; break;
		case 1: v_loc = {1.0, 1.0, 1.0, 1.0}; break;
		case 2: v_loc = {2.0, 2.0, 2.0, 2.0, 2.0}; break;
		default: {
		}
		}
		// gather communication
		std::vector<T> v;
		//  v.reserve(v_local.size()*world.size()); // to avoid _some_ reallocations
		world.all_gatherv(begin(v_loc), end(v_loc), std::back_inserter(v));
		//  v.shrink_to_fit(); // to save _some_ memory

		// check communication
		assert((v==std::vector<T>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0}));
	}
}
