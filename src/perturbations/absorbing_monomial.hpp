/* -*- indent-tabs-mode: t -*- */

#ifndef INQ__PERTURBATIONS__ABSORBING_MONOMIAL
#define INQ__PERTURBATIONS__ABSORBING_MONOMIAL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa, Yifan Yao
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// ----------------------------------------------------------------------------
// NEW (inq-study only): a MONOMIAL complex absorbing potential, the ramp variant
// of perturbations::absorbing. Where the built-in absorbing.hpp is a sin² HUMP
// (zero at both slab edges → no absorption at the box wall), this ramps
//   V(z) = i·amplitude·s^order ,  s = (z − z_inner)/width ∈ [0,1]
// from 0 at the inner edge to |amplitude| at the OUTER (wall) edge — peak
// absorption where the outgoing wave would otherwise reflect. `order` (n) tunes
// the onset smoothness (n=1 linear … larger n = gentler onset, less total
// absorption). amplitude < 0 absorbs (exp(-iVt) = exp(amplitude·s^n·t)).
//
// Form: monomial/polynomial CAP family (De Giovannini, Larsen & Rubio 2014,
// arXiv:1409.1689 §IV; reflection-vs-order theory: Riss & Meyer 1996,
// J. Phys. B 26, 4503). Lives in inq-study because inq/ is immutable and the
// engine ships only the sin² CAP. Relies on the inq-study scalar-potential
// complexification so the imaginary potential actually propagates.
// ----------------------------------------------------------------------------

#include <inq_config.h>

#include <math/vector3.hpp>
#include <magnitude/energy.hpp>
#include <perturbations/none.hpp>

namespace inq {
namespace perturbations {

class absorbing_monomial : public perturbations::none {

public:
	absorbing_monomial(quantity<magnitude::energy> amplitude, double mid_pos, double width, int order):
		amplitude_(amplitude.in_atomic_units()),
		mid_pos_(mid_pos),
		width_(width),
		order_(order)
	{
		mid_pos_ = fmod(mid_pos_, 1); //move mid_pos_ to the first unit cell
		if (mid_pos_ >= 0.5) mid_pos_ = mid_pos_ - 1;
		assert(mid_pos_ >= -0.5 and mid_pos_ < 0.5);
		assert(order_ >= 1);
	}

	auto has_potential() const {
		return true;
	}

	template<typename PotentialType>
	void potential(const double time, PotentialType & potential) const {

		gpu::run(potential.basis().local_sizes()[2], potential.basis().local_sizes()[1], potential.basis().local_sizes()[0],
						 [point_op = potential.basis().point_op(), vk = begin(potential.cubic()), mid_pos = mid_pos_, width = width_, amplitude = amplitude_, order = double(order_)] GPU_LAMBDA (auto iz, auto iy, auto ix) {
							 auto rr = point_op.rvector(ix, iy, iz);
							 if (rr[2] > mid_pos - width/2 and rr[2] < mid_pos + width/2) {
								 auto ss = (rr[2] - (mid_pos - width/2))/width;   // 0 at inner edge, 1 at outer (wall) edge
								 vk[ix][iy][iz] += complex(0.0, amplitude*pow(ss, order));
							 }
						 });
	}

private:
	double amplitude_;
	double mid_pos_;
	double width_;
	int order_;
};

}
}
#endif

#ifdef INQ_PERTURBATIONS_ABSORBING_MONOMIAL_UNIT_TEST
#undef INQ_PERTURBATIONS_ABSORBING_MONOMIAL_UNIT_TEST

#include <catch2/catch_all.hpp>
#include <basis/real_space.hpp>

using namespace inq;
using namespace Catch::literals;
using namespace magnitude;

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
    perturbations::absorbing_monomial nop(1.0_Ha, 0.3, 0.1, 2);
    CHECK(nop.has_potential());
    CHECK(not nop.has_uniform_electric_field());
}
#endif
