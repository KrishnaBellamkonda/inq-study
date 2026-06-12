// Copyright 2022-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <cassert>
#include <chrono>  // NOLINT(misc-include-cleaner) bug in clang-tidy 18.1
#include <iostream>
#include <thread>

namespace mpi3 = boost::mpi3;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	auto const my_rank = world.rank();

	mpi3::request r = world.ibarrier();

	using namespace std::literals::chrono_literals;
	std::this_thread::sleep_for(2s);  // NOLINT(misc-include-cleaner) bug in clang-tidy 18.1

	std::cout<<"mpi process "<< my_rank <<" call ibarrier.\n";
	r.wait();
	assert( r.completed() );
	std::cout<<"mpi process "<< my_rank <<" the barrier is complete.\n";
}
