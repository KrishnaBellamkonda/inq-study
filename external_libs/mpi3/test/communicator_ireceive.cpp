// Copyright 2017-2025 Alfredo A. Correa

#include <mpi3/communicator.hpp>
#include <mpi3/environment.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

namespace mpi3 = boost::mpi3;

using std::cout;

int main(int argc, char** argv) {  // NOLINT(bugprone-exception-escape)
	mpi3::environment env(argc, argv);

	auto world = env.world();

	assert( world.size() > 1 );

	using T = double;
	std::vector<T> const cbuffer = {0, 1, 2};
	std::vector<T> buffer(3); // TODO(correaa): test with list

	int const right = world.right();
	int const left = world.left();
	{
		auto req = world.ireceive_n(buffer.begin(), buffer.size(), left);
		world.send(cbuffer.begin(), cbuffer.end(), right);
		cout <<"waiting ireceive in rank "<< world.rank() << '\n';
		req.wait();
		cout <<"ireceive completed in rank "<< world.rank() << '\n';
	}
	assert( std::equal(cbuffer.begin(), cbuffer.end(), buffer.begin()) );
}
