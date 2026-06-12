/* -*- indent-tabs-mode: t -*- */

#ifndef PSEUDOPOD__MATH__SPHERICAL_BESSEL
#define PSEUDOPOD__MATH__SPHERICAL_BESSEL

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <gpu/run.hpp>

namespace pseudo {
	namespace math {

		GPU_FUNCTION auto spherical_bessel(int const ll, double const xx){

			assert(ll <= 5);

			auto xx2 = xx*xx;
			auto xx3 = xx2*xx;
			auto xx4 = xx2*xx2;
			auto xx5 = xx2*xx3;
			
			switch(ll){
			case 0: {
				if(fabs(xx) < 1e-5) return 1.0 - xx*xx/6.0;
				return sin(xx)/xx;
			}
				
			case 1: {
				if(fabs(xx) < 1e-5) return xx/3.0;
				return (sin(xx)/xx - cos(xx))/xx;
			}
				
			case 2: {
				if(fabs(xx) < 1e-5) return xx*xx/6.0;
				return (3.0/xx2 - 1.0)*(sin(xx)/xx) - 3.0*cos(xx)/xx2;
			}
				
			case 3: {
				if(fabs(xx) < 1e-5) return 0.0;
				return (15.0/xx3 - 6.0/xx)*sin(xx)/xx - (15.0/xx2 - 1.0)*cos(xx)/xx;
			}
				
			case 4: {
				if(fabs(xx) < 1e-5) return 0.0;
				return (105.0/xx4 - 45.0/xx2 + 1)*sin(xx)/xx + (-105.0/xx3 + 10.0/xx)*cos(xx)/xx;
			}

			case 5: {
				if(fabs(xx) < 1e-5) return 0.0;
				return (945.0/xx5 - 420.0/xx3 + 15.0/xx)*sin(xx)/xx + (-945.0/xx4 + 105.0/xx2 - 1.0)*cos(xx)/xx;
			}

			default:
				assert(false);
				return 0.0;
			}		 
			
		}
			
	}
}

#endif

#ifdef INQ_MATH_SPHERICAL_BESSEL_UNIT_TEST
#undef INQ_MATH_SPHERICAL_BESSEL_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	using namespace Catch::literals;
	using Catch::Approx;

	using pseudo::math::spherical_bessel;
	
	CHECK(spherical_bessel(0, 0.00) ==	1.000000000000_a);
	CHECK(spherical_bessel(0, 1e-8) ==	0.9999999999999997_a);
	CHECK(spherical_bessel(0, 0.55) ==	0.9503404162375625_a);
	CHECK(spherical_bessel(0, 0.98) ==	0.8474462964203781_a);
	CHECK(spherical_bessel(0, 1.77) ==	0.5537991371683877_a);
	CHECK(spherical_bessel(0, 2.22) ==	0.3588132757820211_a);
	CHECK(spherical_bessel(0, 3.03) ==	0.0367528675533497_a);
	CHECK(spherical_bessel(0, 4.91) == -0.1997023213429875_a);
	CHECK(spherical_bessel(0, 5.67) == -0.1014947760485392_a);
	CHECK(spherical_bessel(0, 6.16) == -0.0199470771276867_a);
	CHECK(spherical_bessel(0, 7.37) ==	0.1201016636425968_a);
	CHECK(spherical_bessel(0, 8.27) ==	0.1106052298813976_a);
	CHECK(spherical_bessel(0, 9.67) == -0.0251056594987433_a);
	
	CHECK(spherical_bessel(1, 0.00) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(1, 1e-8) ==	0.0000000033333333_a);
	CHECK(spherical_bessel(1, 0.55) ==	0.1778470803237389_a);
	CHECK(spherical_bessel(1, 0.98) ==	0.2963507649532252_a);
	CHECK(spherical_bessel(1, 1.77) ==	0.4246824586313540_a);
	CHECK(spherical_bessel(1, 2.22) ==	0.4339484445225002_a);
	CHECK(spherical_bessel(1, 3.03) ==	0.3401098568224111_a);
	CHECK(spherical_bessel(1, 4.91) == -0.0806577856639033_a);
	CHECK(spherical_bessel(1, 5.67) == -0.1621364867726162_a);
	CHECK(spherical_bessel(1, 6.16) == -0.1643456748163642_a);
	CHECK(spherical_bessel(1, 7.37) == -0.0468392851159859_a);
	CHECK(spherical_bessel(1, 8.27) ==	0.0622402508353313_a);
	CHECK(spherical_bessel(1, 9.67) ==	0.0977226250816984_a);

	CHECK(spherical_bessel(2, 0.00) ==	0.0000000000000000_a);
	CHECK(fabs(spherical_bessel(2, 1e-8)) < 1e-16);
	CHECK(spherical_bessel(2, 0.55) ==	0.0197345673464681_a);
	CHECK(spherical_bessel(2, 0.98) ==	0.0597499228241890_a);
	CHECK(spherical_bessel(2, 1.77) ==	0.1660016401728905_a);
	CHECK(spherical_bessel(2, 2.22) ==	0.2276035411402767_a);
	CHECK(spherical_bessel(2, 3.03) ==	0.2999895649440870_a);
	CHECK(spherical_bessel(2, 4.91) ==	0.1504205785748186_a);
	CHECK(spherical_bessel(2, 5.67) ==	0.0157082751106469_a);
	CHECK(spherical_bessel(2, 6.16) == -0.0600914008672959_a);
	CHECK(spherical_bessel(2, 7.37) == -0.1391678583980863_a);
	CHECK(spherical_bessel(2, 8.27) == -0.0880271461442762_a);
	CHECK(spherical_bessel(2, 9.67) ==	0.0554229165044409_a);

	CHECK(spherical_bessel(3, 0.00) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(3, 1e-8) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(3, 0.55) ==	0.0015580773714256_a);
	CHECK(spherical_bessel(3, 0.98) ==	0.0084957800681473_a);
	CHECK(spherical_bessel(3, 1.77) ==	0.0442487282977151_a);
	CHECK(spherical_bessel(3, 2.22) ==	0.0786721436312759_a);
	CHECK(spherical_bessel(3, 3.03) ==	0.1549224285638710_a);
	CHECK(spherical_bessel(3, 4.91) ==	0.2338355642533315_a);
	CHECK(spherical_bessel(3, 5.67) ==	0.1759885812264495_a);
	CHECK(spherical_bessel(3, 6.16) ==	0.1155701870994033_a);
	CHECK(spherical_bessel(3, 7.37) == -0.0475758155611419_a);
	CHECK(spherical_bessel(3, 8.27) == -0.1154610163397304_a);
	CHECK(spherical_bessel(3, 9.67) == -0.0690654810773338_a);
	
	CHECK(spherical_bessel(4, 0.00) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(4, 1e-8) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(4, 0.55) ==	0.0000955082898582_a);
	CHECK(spherical_bessel(4, 0.98) ==	0.0009342205197200_a);
	CHECK(spherical_bessel(4, 1.77) ==	0.0089933304960394_a);
	CHECK(spherical_bessel(4, 2.22) ==	0.0204617766159988_a);
	CHECK(spherical_bessel(4, 3.03) ==	0.0579170356985193_a);
	CHECK(spherical_bessel(4, 4.91) ==	0.1829498796274871_a);
	CHECK(spherical_bessel(4, 5.67) ==	0.2015615782553401_a);
	CHECK(spherical_bessel(4, 6.16) ==	0.1914211589347996_a);
	CHECK(spherical_bessel(4, 7.37) ==	0.0939805166168118_a);
	CHECK(spherical_bessel(4, 8.27) == -0.0097028555943106_a);
	CHECK(spherical_bessel(4, 9.67) == -0.1054186111829660_a);

	CHECK(spherical_bessel(5, 0.00) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(5, 1e-8) ==	0.0000000000000000_a);
	CHECK(spherical_bessel(5, 0.55) ==	0.0000047855535264_a);
	CHECK(spherical_bessel(5, 0.98) ==	0.0000837961333631_a);
	CHECK(spherical_bessel(5, 1.77) ==	0.0014800708346885_a);
	CHECK(spherical_bessel(5, 2.22) ==	0.0042810048119624_a);
	CHECK(spherical_bessel(5, 3.03) ==	0.0171083705406418_a);
	CHECK(spherical_bessel(5, 4.91) ==	0.1015104472838138_a);
	CHECK(spherical_bessel(5, 5.67) ==	0.1439504318772649_a);
	CHECK(spherical_bessel(5, 6.16) ==	0.1641035840715701_a);
	CHECK(spherical_bessel(5, 7.37) ==	0.1623417110226488_a);
	CHECK(spherical_bessel(5, 8.27) ==	0.1049016813519679_a);
	CHECK(spherical_bessel(5, 9.67) == -0.0290490484621381_a);
	
}
#endif
