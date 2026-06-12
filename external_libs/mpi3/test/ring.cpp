// Copyright 2021-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>
#include <mpi3/process.hpp>

#include <cassert>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	auto const size = world.size();
	assert(size != 0);

	mpi3::process&& next  = world[(world.rank() + size + 1) % size];
	mpi3::process&& prior = world[(world.rank() + size - 1) % size];

	int token;  // NOLINT(cppcoreguidelines-init-variables)
	if(!world.is_root()) {
		prior >> token;
	} else {
		token = -1;
	}

	next << token;

	if(world.is_root()) {
		prior >> token;
	}

	assert(token == -1);
}
