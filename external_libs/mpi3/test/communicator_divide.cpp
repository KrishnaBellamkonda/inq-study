// Copyright 2018-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <cassert>
#include <iostream>

namespace mpi3 = boost::mpi3;

using std::cout;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	assert(world.size() == 6);

	mpi3::communicator const fifth = world / 5;

	cout << "I am rank " << world.rank() << " in " << world.name() << ", ";

	if(fifth) {
		cout << "I am also   " << fifth.rank() << " in " << fifth.name() << '\n';
	} else {
		cout << "I am not in " << fifth.name() << '\n';
	}
}
