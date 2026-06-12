// Copyright 2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

namespace mpi3 = boost::mpi3;  // NOLINT(misc-unused-alias-decls)

template<class F> static auto operator*(F&& f) -> decltype(std::forward<F>(f)()) { return std::forward<F>(f)(); }  // NOLINT(misc-use-anonymous-namespace,readability-static-definition-in-anonymous-namespace)

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);  // NOLINT(misc-const-correctness)

#ifndef _MSC_VER
// #if !defined(__INTEL_MPI__) && !defined(__INTEL_LLVM_COMPILER)
	auto world = env.world();
	{
		auto const local_rank_string = *[rank = world.rank()] {
			switch(rank) {
			case 0 : return std::string{"abc"};
			case 1 : return std::string{"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
			default: return "abcdefghijklmnopqrstuvwxyz|" + std::to_string(rank);
			}
		};
		std::vector<std::string> rank_strings(static_cast<std::size_t>(world.size()));

		// auto end =
		world.all_gather_n(&local_rank_string, 1, rank_strings.begin());

		// assert(end == rank_strings.end());
		assert( world.size() >= 2 );
		assert( rank_strings[0] == std::string{"abc"});
		assert( rank_strings[1] == std::string{"ABCDEFGHIJKLMNOPQRSTUVWXYZ"} );
	}
#endif
}
