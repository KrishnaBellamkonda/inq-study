// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <mpi3/detail/mpi_impl.h>
#include <mpi3/group.hpp>

#include <cassert>
#include <type_traits>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	mpi3::group const        wg{world};
	mpi3::communicator const w2{wg};

	assert(w2.rank() == world.rank());
	assert(w2.size() == world.size());

	mpi3::communicator const half = world / 2;
	mpi3::group const        hg{half};

	mpi3::communicator const h2{hg};
	assert(half.rank() == h2.rank());

	static_assert(std::is_same_v<decltype(&const_cast<mpi3::group&>(wg)), MPI_Group>);  // NOLINT(cppcoreguidelines-pro-type-const-cast) test const_cast

	[[maybe_unused]] mpi3::group const& wgc = wg;
	static_assert(std::is_same_v<decltype(&wgc), mpi3::group const*>);

	// static_assert( std::is_same<decltype(*&wg), mpi3::group&>{}, "!" );
}
