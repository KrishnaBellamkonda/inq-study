/* -*- indent-tabs-mode: t -*- */

#ifndef SHARMONIC
#define SHARMONIC

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.


// THIS FILE IS PART OF THE SHARMONIC LIBRARY
// A FULL DISTRIBUTION OF THE LIBRARY (INCLUDING TESTS)
// CAN BE FOUND AT:
//
//   https://gitlab.com/npneq/sharmonic
//

#ifdef HAVE_SHARMONIC_CONFIG_H
#include <sharmonic_config.h>
#endif

#ifdef __cplusplus
#include <cmath>
#include <complex>
#include <cassert>
#include <array>

#define ComplexConstr(real, imag) Complex{real, imag}

#define SH_FUNC(func) func
#else

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdlib.h>

#define Complex double complex
#define ComplexConstr(real, imag) real + I*imag

#define SH_FUNC(func) sharmonic_##func
#endif


#define SHARMONIC_MAX_L 9

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SH_GPU_FUNCTION __host__ __device__ inline
#else
#define SH_GPU_FUNCTION inline
#endif

#ifdef __cplusplus
namespace sharmonic {
namespace internal {
#endif
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(swap)(double * aa, double  * bb) {
	double cc = *aa;
	*aa = *bb;
	*bb = cc;
};

SH_GPU_FUNCTION static double SH_FUNC(norm)(double xx, double yy, double zz) {
	xx = fabs(xx);
	yy = fabs(yy);
	zz = fabs(zz);

	if (xx < yy) SH_FUNC(swap)(&xx, &yy);
	if (yy < zz) SH_FUNC(swap)(&yy, &zz);
	if (xx < yy) SH_FUNC(swap)(&xx, &yy);

	if(xx == 0.0) return 0.0;
	
	yy /= xx;
	zz /= xx;

	return xx*sqrt(1.0 + yy*yy + zz*zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(normalize)(double * xx, double * yy, double * zz) {
	double nn = SH_FUNC(norm)(*xx, *yy, *zz);
	if(nn == 0.0) return 0.0;
	
  *xx /= nn;
	*yy /= nn;
	*zz /= nn;
	return nn;
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(angular_to_normalized_cartesian)(double theta, double phi, double * xx, double * yy, double * zz) {
	*xx = sin(theta)*cos(phi);
	*yy = sin(theta)*sin(phi);
	*zz = cos(theta);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(normalized_cartesian_to_angular)(double xx, double yy, double zz, double * theta, double * phi) {
	*phi = atan2(yy, xx);
	*theta = acos(zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(cartesian_to_angular)(double xx, double yy, double zz, double * rr, double * theta, double * phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	*rr = SH_FUNC(normalize)(&xx, &yy, &zz);

	if(*rr < 1.0e-15) {
		*theta = 0.0;
		*phi   = 0.0;
		return;
	}

	SH_FUNC(normalized_cartesian_to_angular)(xx, yy, zz, theta, phi);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(powsign)(int mm) {
	if(mm%2 == 0) return 1.0;
	return -1.0;
}

///////////////////////////////////////////////////////////////////

//this functions converts an mm > 0 complex harmonic into the -mm complex harmonic
#ifdef __cplusplus
template <typename Complex>
#endif
SH_GPU_FUNCTION static Complex SH_FUNC(neg)(int mm, double re, double im) {
	if(mm >= 0) return ComplexConstr(re, im);
	if(abs(mm)%2 == 0) return ComplexConstr(re, -im);
	return ComplexConstr(-re, im);
}

SH_GPU_FUNCTION static double rpair(double const a, double const b) {
	return (a - b)*(a + b);
}

#ifdef __cplusplus
} //namespace internal
#endif

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(sincos_legendre)(int const ll, int const mm, double sintheta, double costheta);

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(normalized_cartesian_real)(int const ll, int const mm, double x, double y, double z) {
	assert(ll >= 0);

	//this assertion fails on hip for some reason
#ifndef __HIPCC__ 
	assert(abs(mm) <= ll);
#endif

#ifdef __cplusplus
	using internal::rpair;
	using internal::powsign;
#endif
	
	if(ll > SHARMONIC_MAX_L) {
		double const leg = SH_FUNC(sincos_legendre)(ll, mm, sqrt(rpair(1, z)), z);
		double phi = atan2(y, x);
		
		if(mm > 0){
			return SH_FUNC(powsign)(mm)*sqrt(2.0)*cos(mm*phi)*leg;
		} else if (mm == 0) {
			return leg;
		} else {
			return SH_FUNC(powsign)(mm)*sqrt(2.0)*sin(-mm*phi)*leg;
		}
	}
	
	if(mm%2 == 1) return SH_FUNC(powsign)(mm/2)*SH_FUNC(normalized_cartesian_real)(ll, -mm, -y, x, z); // 90 degree rotation
	if(mm%4 == 2) return SH_FUNC(powsign)(mm/4)*SH_FUNC(normalized_cartesian_real)(ll, -mm, (x - y)/sqrt(2), (x + y)/sqrt(2), z);          // 45 degree rotation
	if(mm > 0) {
		double rotcos, rotsin;
		if(mm == 4) {
			rotcos = cos(M_PI/8.0);
			rotsin = sin(M_PI/8.0);
		} else if(mm == 8) {
			rotcos = cos(M_PI/16.0);
			rotsin = sin(M_PI/16.0);
		} else {
			rotcos = cos(M_PI/(2.0*mm));
			rotsin = sin(M_PI/(2.0*mm));
		}
		return SH_FUNC(normalized_cartesian_real)(ll, -mm, rotcos*x - rotsin*y, rotsin*x + rotcos*y, z);
	}
	
	// Functions taken from:
	//   [1] https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
	//   [2] https://www.doiserbia.nb.rs/img/doi/1450-698X/2009/1450-698X0979107M.pdf
	//   [3] https://github.com/elerac/sh_table/blob/main/docs/table.txt

	switch(ll){
	case 0:
		return 1.0/sqrt(4*M_PI);
	case 1:
		switch(mm){
		case -1: return sqrt(3.0/(4*M_PI))*y;
		case	0: return sqrt(3.0/(4*M_PI))*z;
		default: return 0.0;
		}
	case 2:
		switch(mm){		
		case -2: return 1.0/2.0*sqrt(15/M_PI)*x*y;
		case -1: return 1.0/2.0*sqrt(15/M_PI)*y*z;
		case	0: return 1.0/4.0*sqrt(5/M_PI)*rpair(sqrt(3)*z, 1);
		default: return 0.0;
		}
	case 3:
		switch(mm){		
		case -3: return 1.0/8.0*sqrt(70/M_PI)*y*rpair(sqrt(3)*x, y);
		case -2: return 1.0/2.0*sqrt(105/M_PI)*x*y*z;
		case -1: return 1.0/8.0*sqrt(42/M_PI)*y*rpair(sqrt(5)*z, 1);
		case	0: return 1.0/4.0*sqrt(7/M_PI)*rpair(sqrt(5)*z, sqrt(3))*z;
		default: return 0.0;
		}
	case 4:
		switch(mm){
		case -4: return 3.0/4.0*sqrt(35/M_PI)*x*y*rpair(x, y);
		case -3: return 3.0/8.0*sqrt(70/M_PI)*y*rpair(sqrt(3)*x, y)*z;
		case -2: return 3.0/4.0*sqrt(5/M_PI)*x*y*rpair(sqrt(7)*z, 1);
		case -1: return 3.0/8.0*sqrt(10/M_PI)*y*rpair(sqrt(7)*z, sqrt(3))*z;
		case	0: return 105.0/16.0*sqrt(1/M_PI)*rpair(sqrt((3 + (2*sqrt(6/5.0)))/7.0), z)*rpair(sqrt((3 - (2*sqrt(6/5.0)))/7.0), z);
		default: return 0.0;
		}
	case 5:
		switch(mm){		
		case -5: return 30.0/64.0*sqrt(154/M_PI)*y*rpair(sqrt(1 - 2/sqrt(5))*y, x)*rpair(sqrt(1 + 2/sqrt(5))*y, x);
		case -4: return 3.0/4.0*sqrt(385.0/M_PI)*rpair(x, y)*y*z*x;
		case -3: return 1.0/32.0*sqrt(770.0/M_PI)*rpair(1, 3*z)*rpair(y, sqrt(3)*x)*y;
		case -2: return 1.0/4.0*sqrt(1155.0/M_PI)*y*rpair(sqrt(3)*z, 1)*z*x;
		case -1: return 21.0/16.0*sqrt(165.0/M_PI)*y*rpair(sqrt((7 - 2*sqrt(7))/21), z)*rpair(sqrt((7 + 2*sqrt(7))/21), z);
		case	0: return 63.0/16.0*sqrt(11.0/M_PI)*z*rpair(sqrt(5 - 2*sqrt(10/7.0))/3, z)*rpair(sqrt(5 + 2*sqrt(10/7.0))/3, z);
		default: return 0.0;
		}
	case 6:
		switch(mm){
		case -6: return 1.0/32.0*sqrt(6006/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
		case -5: return 3.0/32*sqrt(2002/M_PI)*y*z*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);			
		case -4: return 3.0/8.0*sqrt(91/M_PI)*x*y*rpair(x, y)*rpair(sqrt(11)*z, 1);
		case -3: return 1.0/32.0*sqrt(2730/M_PI)*y*z*rpair(sqrt(3)*x, y)*rpair(sqrt(11)*z, sqrt(3));
		case -2: return 33.0/32.0*sqrt(2730.0/M_PI)*x*y*rpair(sqrt((3 + 4/sqrt(3))/11), z)*rpair(sqrt((3 - 4/sqrt(3))/11), z);
		case -1: return 33.0/16.0*sqrt(273.0/M_PI)*z*rpair(sqrt((15 - 2*sqrt(15))/33), z)*y*rpair(sqrt((15 + 2*sqrt(15))/33), z);
		case  0: {
			// ANALYTIC: return 1.0/32*sqrt(13.0/M_PI)*(z2*((231*z2 - 315)*z2 + 105) - 5);
			double const z0 = 0.23861918608319690863050172168071194;
			double const z1 = 0.66120938646626451366139959501990535;
			double const z2 = 0.93246951420315202781230155449399461;
			return -231.0/32*sqrt(13.0/M_PI)*rpair(z0, z)*rpair(z1, z)*rpair(z2, z);
		}
		default: return 0.0;
		}
	case 7:
		switch(mm){
		case -7: {
			// ANALYTIC: return 3.0/64.0*sqrt(715/M_PI)*y*(7.0*x6 - 35.0*x4*y2 + 21.0*x2*y4 - y6);
			double const r1 = 0.481574618807528644332162353056970575219;
			double const r2 = 1.253960337662703837570910978336464443221;
			double const r3 = 4.3812862675348230724046890850326954441502;
			return 3.0/64.0*sqrt(715/M_PI)*y*rpair(r1*x, y)*rpair(r2*x, y)*rpair(r3*x, y);
		}
		case -6: return 3.0/32.0*sqrt(10010/M_PI)*x*y*z*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
		case -5: return 3.0/64.0*sqrt(385/M_PI)*y*rpair(sqrt(13)*z, 1)*rpair(sqrt(5 - 2*sqrt(5))*x,y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
		case -4: return 3.0/8.0*sqrt(385/M_PI)*x*y*z*rpair(x, y)*rpair(sqrt(13.0)*z, sqrt(3));
		case -3: return 429.0/64.0*sqrt(35/M_PI)*y*rpair(sqrt(3)*x, y)*rpair(sqrt((33 - 2*sqrt(165))/143), z)*rpair(sqrt((33 + 2*sqrt(165))/143), z);
		case -2: return 429.0/32.0*sqrt(70/M_PI)*x*y*z*rpair(sqrt((55 - 4*sqrt(55))/143), z)*rpair(sqrt((55 + 4*sqrt(55))/143), z);
		case -1: {
			// ANALYTIC:	return 1.0/64.0*sqrt(105/M_PI)*y*(z2*((429*z2 - 495)*z2 + 135) - 5);
			double const r1 = 0.20929921790247886876865726034535126;
			double const r2 = 0.59170018143314230214451073139795319;
			double const r3 = 0.87174014850960661533744576122066344;
			return -429/64.0*sqrt(105/M_PI)*y*rpair(r1, z)*rpair(r2, z)*rpair(r3, z);
		}
		case  0: {
			// ANALYTIC: return 1.0/32.0*sqrt(15/M_PI)*z*(-35 + z2*(315 + z2*(-693 + 429*z2)));
			double const z0 = 0.40584515137739716690660641207696146;
			double const z1 = 0.74153118559939443986386477328078841;
			double const z2 = 0.94910791234275852452618968404785126;
			return -429/32.0*sqrt(15/M_PI)*z*rpair(z0, z)*rpair(z1, z)*rpair(z2, z);
		}
		default: return 0.0;
		}
	case 8:
		switch(mm){
		case -8: return 12.0/128.0*sqrt(12155/M_PI)*x*y*rpair(x, y)*rpair((1 + sqrt(2))*x, y)*rpair((1 - sqrt(2))*x, y);
		case -7: {
			// ANALYTIC: return 3.0/64.0*sqrt(12155/M_PI)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
			double const r1 = 0.481574618807528644332162353056970575219;
			double const r2 = 1.253960337662703837570910978336464443221;
			double const r3 = 4.381286267534823072404689085032695444150;
			return 3.0/64.0*sqrt(12155/M_PI)*y*z*rpair(r1*x, y)*rpair(r2*x, y)*rpair(r3*x, y);
		}
		case -6: return 1.0/64.0*sqrt(14586/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y)*rpair(sqrt(15)*z, 1);
		case -5: return 3.0/64.0*sqrt(17017/M_PI)*y*z*rpair(sqrt(5)*z, 1)*rpair(sqrt(5 + 2*sqrt(5))*x, y)*rpair(sqrt(5 - 2*sqrt(5))*x, y);
		case -4: return 195.0/32.0*sqrt(1309/M_PI)*x*y*rpair(x, y)*rpair(sqrt((13 - 2*sqrt(26))/65), z)*rpair(sqrt((13 + 2*sqrt(26))/65), z);
		case -3: return 39.0/64.0*sqrt(19635/M_PI)*y*z*rpair(sqrt(3)*x, y)*rpair(sqrt((1 - 2/sqrt(13))/3), z)*rpair(sqrt((1 + 2/sqrt(13))/3), z);
		case -2: {
			// ANALYTIC: return 3.0/64.0*sqrt(1190/M_PI)*x*y*(z2*((143*z2 - 143)*z2 + 33) - 1);
			double const z0 = 0.18867742249078593782624756431860912;
			double const z1 = 0.54060463738735870397940596741354015;
			double const z2 = 0.81984599546348695481600203138243347;
			return -429.0/64.0*sqrt(1190/M_PI)*x*y*rpair(z0, z)*rpair(z1, z)*rpair(z2, z);
		}
		case -1: {
			// ANALYTIC: return 3.0/64.0*sqrt(17/M_PI)*y*z*(z2*((715*z2 - 1001)*z2 + 385) - 35);
			double const z0 = 0.36311746382617815871075206870865921;
			double const z1 = 0.67718627951073775344588542709134245;
			double const z2 = 0.89975799541146015731234524441833796;
			return 2145.0/64.0*sqrt(17/M_PI)*y*z*rpair(z, z0)*rpair(z, z1)*rpair(z, z2);
		}
		case  0: {
			// ANALYTIC: return 1.0/256.0*sqrt(17/M_PI)*((z2*((6435*z2 - 12012)*z2 + 6930) - 1260)*z2 + 35);
			double const z0 = 0.18343464249564980493947614236018398;
			double const z1 = 0.52553240991632898581773904918924635;
			double const z2 = 0.79666647741362673959155393647583044;
			double const z3 = 0.96028985649753623168356086856947299;
			return 6435.0/256.0*sqrt(17/M_PI)*rpair(z0, z)*rpair(z1, z)*rpair(z2, z)*rpair(z3, z);
		}
		default: return 0.0;
		}
	case 9:
		switch(mm){		
		case -9: {
			// ANALYTIC: return 1.0/512.0*sqrt(461890/M_PI)*y*(sqrt(3)*x - y)*(sqrt(3)*x + y)*(3*x6 - 27*x4*y2 + 33*x2*y4 - y6);
			double const r1 = 0.363970234266202361351047882776834043890;
			double const r2 = 0.839099631177280011763127298123181364687;
			double const r3 = 5.6712818196177095309944184398639644216254;
			return 1.0/512.0*sqrt(461890/M_PI)*y*rpair(y, sqrt(3)*x)*rpair(y, r1*x)*rpair(y, r2*x)*rpair(y, r3*x);
		}
		case -8: return 3.0/32.0*sqrt(230945/M_PI)*x*y*z*rpair(x, y)*rpair((1 + sqrt(2))*x, y)*rpair((1 - sqrt(2))*x, y);
		case -7: {
			// ANALYTIC: return -3.0/512.0*sqrt(27170/M_PI)*y*(1 - sqrt(17)*z)*(1 + sqrt(17)*z)*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
			double const r1 = 0.481574618807528644332162353056970575219;
			double const r2 = 1.253960337662703837570910978336464443221;
			double const r3 = 4.3812862675348230724046890850326954441502;
			return 3.0/512.0*sqrt(27170/M_PI)*y*rpair(1, sqrt(17)*z)*rpair(y, r1*x)*rpair(y, r2*x)*rpair(y, r3*x);
		}
		case -6: return 1.0/64.0*sqrt(81510/M_PI)*x*y*z*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y)*rpair(sqrt(17)*z, sqrt(3));
		case -5: return 255.0/256.0*sqrt(5434/M_PI)*y*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y)*rpair(sqrt((15 - 2*sqrt(35))/85), z)*rpair(sqrt((15 + 2*sqrt(35))/85), z);
		case -4: return 51.0/32.0*sqrt(95095/M_PI)*x*y*z*rpair(x, y)*rpair(sqrt((5 - 2*sqrt(2))/17), z)*rpair(sqrt((5 + 2*sqrt(2))/17), z);
		case -3: {
			// ANALYTIC: return 1.0/256.0*sqrt(43890/M_PI)*y*rpair(sqrt(3)*x, y)*(z2*((221*z2 - 195)*z2 + 39) - 1);
			double const z0 = 0.17315675391810856488617760490859758;
			double const z1 = 0.50091349321188556086106367560187088;
			double const z2 = 0.77553552598657945718567606578088618;
			return -221.0/256.0*sqrt(43890/M_PI)*y*rpair(sqrt(3)*x, y)*rpair(z0, z)*rpair(z1, z)*rpair(z2, z);
		}
		case -2: {
			// ANALYTIC: return -3.0/64.0*sqrt(2090/M_PI)*x*y*z*((z2*(273 - 221*z2) - 91)*z2 + 7);
			double const z0 = 0.33162955360278980220085003507847223;
			double const z1 = 0.62737813976261871800246540074189546;
			double const z2 = 0.85540202628487456905775301621747783;
			return -663/64.0*sqrt(2090/M_PI)*x*y*z*rpair(z0, z)*rpair(z1, z)*rpair(z2, z);
		}
		case -1: {
			// ANALYTIC:   return 3.0/256.0*sqrt(95/M_PI)*y*((z2*((2431*z2 - 4004)*z2 + 2002) - 308)*z2 + 7);
			double const z0 = 0.165278957666387024626219765958173533231150344;
			double const z1 = 0.477924949810444495661175092731257997886772893;
			double const z2 = 0.738773865105505075003106174859830725016185101;
			double const z3 = 0.919533908166458813828932660822338134153543075;
			return 7293.0/256.0*sqrt(95/M_PI)*y*rpair(z0, z)*rpair(z1, z)*rpair(z2, z)*rpair(z3, z);
		}
		case  0: {
			// ANALYTIC: return 1.0/256.0*sqrt(19/M_PI)*z*((z2*(rpair(sqrt(12155)*z, sqrt(25740))*z2 + 18018) - 4620)*z2 + 315);
			double const z0 = 0.32425342340380892903853801464333661;
			double const z1 = 0.61337143270059039730870203934147418;
			double const z2 = 0.83603110732663579429942978806973488;
			double const z3 = 0.96816023950762608983557620290367287;
			return 12155.0/256.0*sqrt(19/M_PI)*z*rpair(z0, z)*rpair(z1, z)*rpair(z2, z)*rpair(z3, z);
		}
		default: return 0.0;
		}
	default: return 0.0;
	}
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(cartesian_real)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(norm < 1.0e-15) return 0.0;
	return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(angular_real)(int const ll, int const mm, double theta, double phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);

	double xx, yy, zz;
	SH_FUNC(angular_to_normalized_cartesian)(theta, phi, &xx, &yy, &zz);
	return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);;
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION static Complex SH_FUNC(normalized_cartesian_complex)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	if(mm == 0) return SH_FUNC(normalized_cartesian_real)(ll, mm, xx, yy, zz);
	double factor = SH_FUNC(powsign)(abs(mm))*0.5*sqrt(2.0);
	double re = factor*SH_FUNC(normalized_cartesian_real)(ll, abs(mm), xx, yy, zz);
	double im = factor*SH_FUNC(normalized_cartesian_real)(ll, -abs(mm), xx, yy, zz);

#ifdef __cplusplus
	return SH_FUNC(neg)<Complex>(mm, re, im);
#else
	return SH_FUNC(neg)(mm, re, im);
#endif
	
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION static Complex SH_FUNC(cartesian_complex)(int const ll, int const mm, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);

	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(norm < 1.0e-15) return 0.0;

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_complex)(ll, mm, xx, yy, zz);
#endif
	
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
#endif
SH_GPU_FUNCTION static Complex SH_FUNC(angular_complex)(int const ll, int const mm, double theta, double phi) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double xx, yy, zz;
	SH_FUNC(angular_to_normalized_cartesian)(theta, phi, &xx, &yy, &zz);

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_complex)(ll, mm, xx, yy, zz);
#endif
}

#ifdef __cplusplus

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus

template <typename Complex>
#endif
struct spinor {
#ifdef __cplusplus
	SH_GPU_FUNCTION spinor(Complex const & s0, Complex const & s1){
		components[0] = s0;
		components[1] = s1;
	}
	SH_GPU_FUNCTION auto & operator[](int index) const{
		return components[index];
	}
	SH_GPU_FUNCTION auto & operator[](int index){
		return components[index];
	}
private:
#endif	
	Complex components[2];
};


///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
auto
#else
spinor
#endif
SH_GPU_FUNCTION static SH_FUNC(normalized_cartesian_spinor)(int const twice_j, int const twice_mj, int const twice_s, double xx, double yy, double zz) {

	// These come from https://en.wikipedia.org/wiki/Spinor_spherical_harmonics
	// we use that '2*ll = twice_jj - sgn' to simplify the expressions
	assert(twice_j%2 != 0);
	assert(twice_mj%2 != 0);
	assert(abs(twice_mj) <= twice_j);
	assert(abs(twice_s) == 1);
	
	int ll = (twice_j - twice_s)/2;
	double sgn = twice_j - 2*ll;
	double den = 2*ll + 1;
	double cc0 = sgn*sqrt((ll + sgn*twice_mj/2.0 + 0.5)/den);
	double cc1 =     sqrt((ll - sgn*twice_mj/2.0 + 0.5)/den);
	int ml0 = (twice_mj - 1)/2;
	int ml1 = (twice_mj + 1)/2;

	Complex s0 = 0.0;
	Complex s1 = 0.0;
	if(abs(ml0) <= ll) s0 = cc0*sharmonic::cartesian_complex<Complex>(ll, ml0, xx, yy, zz);
	if(abs(ml1) <= ll) s1 = cc1*sharmonic::cartesian_complex<Complex>(ll, ml1, xx, yy, zz);
	
	return spinor<Complex>{s0, s1};
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
template <typename Complex = std::complex<double>>
auto
#else
spinor
#endif
SH_GPU_FUNCTION static SH_FUNC(cartesian_spinor)(int const twice_j, int const twice_mj, int const twice_s, double xx, double yy, double zz) {

#ifdef __cplusplus
	using namespace internal;
#endif

	int ll = (twice_j - twice_s)/2;
	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	if(ll != 0 and norm < 1.0e-15) return spinor<Complex>{0.0, 0.0};

#ifdef __cplusplus
	return SH_FUNC(normalized_cartesian_spinor)<Complex>(twice_j, twice_mj, twice_s, xx, yy, zz);
#else
	return SH_FUNC(normalized_cartesian_spinor)(twice_j, twice_mj, twice_s, xx, yy, zz);
#endif
	
}

///////////////////////////////////////////////////////////////////

template <typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(normalized_cartesian_real)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(normalized_cartesian_real)(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(cartesian_real)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(cartesian_real)(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>, typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(normalized_cartesian_complex)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(normalized_cartesian_complex)<Complex>(ll, mm, vec3[0], vec3[1], vec3[2]);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>, typename Vector3 = std::array<double, 3>>
SH_GPU_FUNCTION auto SH_FUNC(cartesian_complex)(int const ll, int const mm, Vector3 const & vec3) {
	return SH_FUNC(cartesian_complex)<Complex>(ll, mm, vec3[0], vec3[1], vec3[2]);
}
///////////////////////////////////////////////////////////////////
#endif

///////////////////////////////////////////////////////////////////
//
// LEGENDRE
//
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(sincos_legendre)(int const ll, int const mm, double sintheta, double costheta) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	
	if(ll <= SHARMONIC_MAX_L) {
		
		//call the spherical harmonic functions that does the calculation explicitly
		double factor = SH_FUNC(powsign)(abs(mm))*0.5*sqrt(2.0);
		if(mm == 0) factor = 1.0;
		return factor*SH_FUNC(normalized_cartesian_real)(ll, abs(mm), sintheta, 0.0, costheta);

	} else {
		
		//use recurrence
		if(abs(mm) <= ll - 2) {
			double alm = sqrt(((2*ll - 1)*(2*ll + 1.0))/((ll - mm)*(ll + mm)));
			double blm = sqrt(((2*ll + 1)*(ll + mm - 1)*(ll - mm - 1))/((2*ll - 3.0)*(ll + mm)*(ll - mm)));
			return alm*costheta*SH_FUNC(sincos_legendre)(ll - 1, mm, sintheta, costheta) - blm*SH_FUNC(sincos_legendre)(ll - 2, mm, sintheta, costheta);
		}

		if(mm == -ll    ) return -sintheta*sqrt((2*mm - 1.0)/(2.0*mm))*SH_FUNC(sincos_legendre)(ll - 1, mm + 1, sintheta, costheta);
		if(mm == -ll + 1) return  costheta*sqrt(-2*mm + 3.0)          *SH_FUNC(sincos_legendre)(ll - 1, mm,     sintheta, costheta);
		if(mm ==  ll - 1) return  costheta*sqrt( 2*mm + 3.0)          *SH_FUNC(sincos_legendre)(ll - 1, mm,     sintheta, costheta);
		if(mm ==  ll    ) return -sintheta*sqrt((2*mm + 1.0)/(2.0*mm))*SH_FUNC(sincos_legendre)(ll - 1, mm - 1, sintheta, costheta);
		
		return 0.0;
	}
}
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(cos_legendre)(int const ll, int const mm, double costheta) {
	#ifdef __cplusplus
	using internal::rpair;
	#endif
	return SH_FUNC(sincos_legendre)(ll, mm, sqrt(rpair(1, costheta)), costheta);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static double SH_FUNC(angular_legendre)(int const ll, int const mm, double theta) {
	return SH_FUNC(sincos_legendre)(ll, mm, sin(theta), cos(theta));
}

///////////////////////////////////////////////////////////////////
//
// HESSIAN STUFF
//
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(normalized_cartesian_real_r2hessian)(int const ll, int const mm, double x, double y, double z,
																																				  double * gx, double * gy, double * gz, double * hxx, double * hyy, double * hzz, double * hxy, double * hxz, double * hyz) {
	assert(ll >= 0);
	assert(ll <= SHARMONIC_MAX_L);
	//this assertion fails on hip for some reason
#ifndef __HIPCC__
	assert(abs(mm) <= ll);
#endif

#ifdef __cplusplus
	using internal::rpair;
	using internal::powsign;
#endif

	int calc_hessian = (hxx != NULL);
	
	if(mm%2 == 1) {
		if(calc_hessian) { // 90 degree rotation
			SH_FUNC(normalized_cartesian_real_r2hessian)(ll, -mm, -y, x, z, gy, gx, gz, hyy, hxx, hzz, hxy, hyz, hxz);
		} else {
			SH_FUNC(normalized_cartesian_real_r2hessian)(ll, -mm, -y, x, z, gy, gx, gz, NULL, NULL, NULL, NULL, NULL, NULL);
		}
		*gx  *=  SH_FUNC(powsign)(mm/2);
		*gy  *= -SH_FUNC(powsign)(mm/2);
		*gz  *=  SH_FUNC(powsign)(mm/2);
		if(calc_hessian) {
			*hxx *=  SH_FUNC(powsign)(mm/2);
			*hyy *=  SH_FUNC(powsign)(mm/2);
			*hzz *=  SH_FUNC(powsign)(mm/2);
			*hxy *= -SH_FUNC(powsign)(mm/2);
			*hxz *=  SH_FUNC(powsign)(mm/2);
			*hyz *= -SH_FUNC(powsign)(mm/2);
		}
		return;
	}

	if(mm > 0) {
		double rotcos, rotsin;
		if(mm == 2) {
			// pi/4 rotation
			rotcos = 1/sqrt(2);
			rotsin = 1/sqrt(2);
		} else if(mm == 4) {
			rotcos = cos(M_PI/8.0);
			rotsin = sin(M_PI/8.0);
		} else if(mm == 4) {
			rotcos = cos(M_PI/12.0);
			rotsin = sin(M_PI/12.0);			
		} else if(mm == 8) {
			rotcos = cos(M_PI/16.0);
			rotsin = sin(M_PI/16.0);
		} else {
			rotcos = cos(M_PI/(2.0*mm));
			rotsin = sin(M_PI/(2.0*mm));
		}

		double const rotcos2   = rotcos*rotcos;
		double const rotsin2   = rotsin*rotsin;
		double const rotsincos = rotsin*rotcos;
		
		double rgx, rgy, rhxx, rhyy, rhxy, rhxz, rhyz;
		if(calc_hessian) {
			SH_FUNC(normalized_cartesian_real_r2hessian)(ll, -mm, rotcos*x - rotsin*y, rotsin*x + rotcos*y, z, &rgx, &rgy, gz, &rhxx, &rhyy, hzz, &rhxy, &rhxz, &rhyz);
		} else {
			SH_FUNC(normalized_cartesian_real_r2hessian)(ll, -mm, rotcos*x - rotsin*y, rotsin*x + rotcos*y, z, &rgx, &rgy, gz, NULL, NULL, NULL, NULL, NULL, NULL);
		}
		*gx  =  rotcos*rgx + rotsin*rgy;
		*gy  = -rotsin*rgx + rotcos*rgy;
		if(calc_hessian) {
			*hxx =  rotcos2*rhxx + rotsin2*rhyy + 2*rotsincos*rhxy;
			*hyy =  rotsin2*rhxx + rotcos2*rhyy - 2*rotsincos*rhxy;
			*hxy =  (rotcos2 - rotsin2)*rhxy + rotsincos*(rhyy - rhxx);
			*hxz =  rotcos*rhxz + rotsin*rhyz;
			*hyz = -rotsin*rhxz + rotcos*rhyz;
		}
		return;
	}

	double d1x = 0.0;
	double d1y = 0.0;
	double d1z = 0.0;

	double d2xx = 0.0;
	double d2yy = 0.0;
	double d2zz = 0.0;
	double d2xy = 0.0;
	double d2xz = 0.0;
	double d2yz = 0.0;

	double x2 = x*x;
	double x3 = x2*x;
	double x4 = x2*x2;
	double x6 = x3*x3;
	double x8 = x4*x4;
	
	double y2 = y*y;
	double y3 = y2*y;
	double y4 = y2*y2;
	double y6 = y3*y3;
	double y8 = y4*y4;	

	double z2 = z*z;

	double pi = M_PI;
		
	switch(ll){
	case /* l = */ 0:
		break;
	case /* l = */ 1:
    switch(mm){
		case -1:
			d1y = sqrt(3.0/(4*M_PI));
			break;
		case  0:
			d1z = sqrt(3.0/(4*M_PI));
			break;
		default: break;
		}
		break;
  case /* l = */ 2:
    switch(mm){
		case -2:
			d2xy = 1/2.0*sqrt(15/M_PI);
			d1x  = d2xy*y;
			d1y  = d2xy*x;
			break;
		case -1:
			d2yz = 1/2.0*sqrt(15/M_PI);
			d1y  = d2yz*z;
			d1z  = d2yz*y;
			break;
		case 0:
			d2zz = 3/2.0*sqrt(5/M_PI);			
			d1z  = d2zz*z;
			break;
		default: break;
		}
		break;
  case /* l = */ 3:
    switch(mm){
		case -3:
			d2xx =  3/4.0*sqrt(70/M_PI)*y;
			d1x  =  d2xx*x;
			d1y  =  3/8.0*sqrt(70/M_PI)*rpair(x, y);
			d2yy = -d2xx;
			d2xy =  3/4.0*sqrt(70/M_PI)*x;
			break;
		case -2:
			d2xy = 1/2.0*sqrt(105/M_PI)*z;
			d2xz = 1/2.0*sqrt(105/M_PI)*y;
			d2yz = 1/2.0*sqrt(105/M_PI)*x;
			d1x  = d2xy*y;
			d1y  = d2yz*z;
			d1z  = d2xz*x;
			break;
		case -1:
			d2zz = 5/4.0*sqrt(42/M_PI)*y;
			d1y  = 1/8.0*sqrt(42/M_PI)*rpair(sqrt(5)*z, 1);
			d1z  = d2zz*z;
			if(calc_hessian) d2yz = 5/4.0*sqrt(42/M_PI)*z;
			break;
		case 0:
			d1z  = 3/4.0*sqrt(7/M_PI)*rpair(sqrt(5)*z, 1);			
			if(calc_hessian) d2zz = 15/2.0*sqrt(7/M_PI)*z;
			break;
		default: break;
		}
		break;
  case /* l = */ 4:
    switch(mm){
		case -4:
			d1x  =  3/4.0*sqrt(35/M_PI)*y*rpair(sqrt(3)*x, y);
			d1y  =  3/4.0*sqrt(35/M_PI)*x*rpair(x, sqrt(3)*y);
			if(calc_hessian) {
				d2xx =  9/2.0*sqrt(35/M_PI)*x*y;
				d2yy = -d2xx;
				d2xy =  9/4.0*sqrt(35/M_PI)*rpair(x, y);
			}
			break;
		case -3:
			d2xx =  9/4.0*sqrt(70/M_PI)*y*z;
			d2yz =  9/8.0*sqrt(70/M_PI)*rpair(x, y);
			d1x  =  d2xx*x;
			d1y  =  d2yz*z;
			d1z  =  3/8.0*sqrt(70/M_PI)*y*rpair(sqrt(3)*x, y);
			if(calc_hessian) {
				d2yy = -d2xx;
				d2xy =  9/4.0*sqrt(70/M_PI)*x*z;
				d2xz =  9/4.0*sqrt(70/M_PI)*x*y;
			}
			break;
		case -2:
			{
				double const coeff = 21/2.0*sqrt(5/M_PI);
				d2xy = 3/4.0*sqrt(5/M_PI)*rpair(sqrt(7)*z, 1);
				d2zz = coeff*x*y;
				d1x  = d2xy*y;
				d1y  = d2xy*x;
				d1z  = coeff*x*y*z;	
				d1z  = d2zz*z;
				if(calc_hessian){
					d2xz = coeff*y*z;
					d2yz = coeff*x*z;
				}
			}
			break;
		case -1:
			d2yz = 9/8.0*sqrt(10/M_PI)*rpair(sqrt(7)*z, sqrt(1));			
			d1y  = 3/8.0*sqrt(10/M_PI)*rpair(sqrt(7)*z, sqrt(3))*z;
			d1z  = d2yz*y;
			if(calc_hessian) d2zz = 63/4.0*sqrt(10/M_PI)*y*z;
			break;
		case 0:
			d1z  = 15/4.0*sqrt(1/M_PI)*z*rpair(sqrt(7)*z, sqrt(3));
			if(calc_hessian) d2zz = 45/4.0*sqrt(1/M_PI)*rpair(sqrt(7)*z, 1);
			break;
		default: break;
		}
		break;
	case /* l = */ 5:
    switch(mm){
		case -5:
			d1x  =  15/8.0*sqrt(154/M_PI)*x*y*rpair(x, y);
			d1y  =  15/32.0*sqrt(154/M_PI)*rpair((sqrt(2) - 1)*x, y)*rpair((sqrt(2) + 1)*x, y);
			if(calc_hessian) {
				d2xx =  15/8.0*sqrt(154/M_PI)*y*rpair(sqrt(3)*x, y);
				d2yy = -d2xx;
				d2xy =  15/8.0*sqrt(154/M_PI)*x*rpair(x, sqrt(3)*y);
			}
			break;
		case -4:
			{
				double pol = 3/4.0*sqrt(385.0/M_PI)*rpair(x, y);
				d2xz =  3/4.0*sqrt(385/M_PI)*y*rpair(sqrt(3)*x, y);
				d2yz =  3/4.0*sqrt(385/M_PI)*x*rpair(x, sqrt(3)*y);
				d1x  =  d2xz*z;
				d1y  =  d2yz*z;
				d1z  =  pol*x*y;
				if(calc_hessian) {
					d2xx =  9/2.0*sqrt(385/M_PI)*x*y*z;
					d2yy = -d2xx;
					d2xy =  3*pol*z;
				}
			}
			break;
		case -3:
			{
				double pol = 3/16.0*sqrt(770/M_PI)*rpair(3*z, 1);
				d2xy =  pol*x;				
				d2zz =  9/16.0*sqrt(770/M_PI)*y*rpair(sqrt(3)*x, y);			
				d1x  =  d2xy*y;
				d1y  =  0.5*pol*rpair(x, y);				
				d1z  =  d2zz*z;
				if(calc_hessian) {
					d2xx =  pol*y;
					d2yy = -d2xx;
					d2xz =  27/8.0*sqrt(770/M_PI)*x*y*z;
					d2yz =  27/16.0*sqrt(770/M_PI)*z*rpair(x, y);
				}
			}
			break;
		case -2:
			d2xz =  1/4.0*sqrt(1155/M_PI)*(3*rpair(z, sqrt(2)*x) - 1)*y;
			d2yz =  1/4.0*sqrt(1155/M_PI)*(3*rpair(z, sqrt(2)*y) - 1)*x;
			d1x  =  d2xz*z;
			d1y  =  d2yz*z;
			d1z  =  1.0/4.0*sqrt(1155.0/M_PI)*x*y*rpair(sqrt(3)*z, 1.0);
			if(calc_hessian) {
				d2xx = -9/2.0*sqrt(1155/M_PI)*x*y*z;
				d2yy =  d2xx;
				d2xy = -1/4.0*sqrt(1155/M_PI)*z*rpair(sqrt(7), 3*z);
			}
			break;
		case -1:
			d1x = 7.0/4.0*sqrt(165.0/M_PI)*x*y*rpair(1, sqrt(3)*z);
			d1y = 1.0/48.0*sqrt(165/M_PI)*(7*rpair(1, sqrt(3)*z)*(1 + 3*rpair(2*y, z)) - 4);
			if(calc_hessian) {
				d2xx = 7/4.0*sqrt(165/M_PI)*y*(3*rpair(sqrt(2)*x, z) + 1);			
				d2yy = 21/4.0*sqrt(165/M_PI)*y*(rpair(sqrt(2)*y, sqrt(3)*z) + 1);
				d2xy = 7/4.0*sqrt(165/M_PI)*x*(3*rpair(sqrt(2)*y, z) + 1);
			}
			break;
		case 0:
			{
				double const pol = 7/4.0*sqrt(11.0/M_PI)*rpair(sqrt(5), 3*z);
				d1x  = pol*x*z;
				d1y  = pol*y*z;
				d1z  = 63.0/16.0*sqrt(11/M_PI)*rpair(sqrt(5 - 2*sqrt(10/7.0))/3.0, z)*rpair(sqrt(5 + 2*sqrt(10/7.0))/3.0, z);
				d2xx = 7/4.0*sqrt(11/M_PI)*z*(9*rpair(sqrt(2)*x, z) + 5);
				d2yy = 7/4.0*sqrt(11/M_PI)*z*(9*rpair(sqrt(2)*y, z) + 5);
				d2xy = 63/2.0*sqrt(11/M_PI)*x*y*z;
				d2xz = pol*x;
				d2yz = pol*y;
			}
			break;
		default: break;
		}
		break;
	case /* l = */ 6:
    switch(mm){
		case -6:
			{
				d1x = 3/32.0*sqrt(6006/M_PI)*y*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
				d1y = 3/32.0*sqrt(6006/M_PI)*x*rpair(sqrt(5 - 2*sqrt(5))*y, x)*rpair(sqrt(5 + 2*sqrt(5))*y, x);
				if(calc_hessian) {
					d2xx =  15/8.0*sqrt(6006/M_PI)*x*y*rpair(x, y);
					d2yy = -d2xx;
					d2xy =  15/32.0*sqrt(6006/M_PI)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
				}
			}
			break;
		case -5:
			d2xz = 15/8.0*sqrt(2002.0/M_PI)*x*y*rpair(x, y);
			d2yz = 15/32.0*sqrt(2002/M_PI)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);			
			d1x  = d2xz*z;
			d1y  = d2yz*z;
			d1z  = 3/32.0*sqrt(2002/M_PI)*y*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
			if(calc_hessian) {
				d2xx =  15/8.0*sqrt(2002/M_PI)*y*z*rpair(sqrt(3)*x, y);
				d2yy = -d2xx;
				d2xy =  15/8.0*sqrt(2002/M_PI)*x*z*rpair(x, sqrt(3)*y);
			}
			break;
		case -4:
			d1x = -33.0/88.0*sqrt(91.0/M_PI)*y*(5*x2*rpair(sqrt(11)*x, sqrt(6)) - y2*rpair(sqrt(11)*y, sqrt(10)));
			d1y =  33.0/88.0*sqrt(91.0/M_PI)*x*(5*y2*rpair(sqrt(11)*y, sqrt(6)) - x2*rpair(sqrt(11)*x, sqrt(10)));			
			if(calc_hessian) {
				d2xx = -15/2.0*sqrt(91/M_PI)*x*y*rpair(sqrt(11)*x, sqrt(3));
				d2yy =  15/2.0*sqrt(91/M_PI)*x*y*rpair(sqrt(11)*y, sqrt(3));			
				d2xy =  15/8.0*sqrt(91/M_PI)*rpair(x, y)*rpair(sqrt(11)*z, sqrt(5));
			}
			break;
		case -3:
			{
				d1x =  1.0/8.0*sqrt(2730/M_PI)*x*y*z*(1 - 11*rpair(sqrt(2)*x, z));
				d1y =  1/32.0*sqrt(2730/M_PI)*z*(8*y2*rpair(sqrt(11)*y, sqrt(6)) + 3*rpair(1, z)*rpair(sqrt(11)*z, sqrt(3)));
				d1z = -1/32.0*sqrt(2730/M_PI)*y*rpair(sqrt(3)*x, y)*rpair(sqrt(3), sqrt(11)*z);			
				if(calc_hessian) {
					d2xx = -1/8.0*sqrt(2730/M_PI)*y*z*(99*x2 + 11*y2 - 12);
					d2yy = -1/8.0*sqrt(2730/M_PI)*y*z*(33*x2 - 55*y2 + 12);
					d2xy =  3/8.0*sqrt(2730/M_PI)*x*z*rpair(sqrt(11)*z, sqrt(7));
					d2xz = -1.0/8.0*sqrt(2730/M_PI)*x*y*(33*x2 + 11*y2 - 12);
					d2yz =  1.0/32.0*sqrt(2730/M_PI)*(x2*(24 - 33*(x2 + 2*y2)) + y2*(55*y2 - 24));
				}
				}
			break;
		case -2:
			//			f    = 0.0078125*sqrt(2730)*x*y*(132.0*x4 + 264.0*x2*y2 - 192.0*x2 + 132.0*y4 - 192.0*y2 + 64.0)/sqrt(pi); //TODO
			d1x = 1/32.0*sqrt(2730/M_PI)*y*(3*rpair(sqrt(3/11.0), z)*(11*rpair(2*x, z) + 3) - 16/11.0);
			d1y = 1/32.0*sqrt(2730/M_PI)*x*(3*rpair(sqrt(3/11.0), z)*(11*rpair(2*y, z) + 3) - 16/11.0);
			
			d2xx = sqrt(2730)*x*y*(20.625*x2 + 12.375*y2 - 9.0)/sqrt(pi); //TODO
			d2yy = sqrt(2730)*x*y*(12.375*x2 + 20.625*y2 - 9.0)/sqrt(pi); //TODO
			d2xy = sqrt(2730)*(5.15625*x4 + 18.5625*x2*y2 - 9/2.0*x2 + 5.15625*y4 - 9/2.0*y2 + 1/2.0)/sqrt(pi); //TODO
			break;
		case -1:
			//			f    = 1/16.0*sqrt(273)*y*z*(33*x4 + 66*x2*y2 - 36*x2 + 33*y4 - 36*y2 + 8)/sqrt(pi); //TODO
			d1x = 3/4.0*sqrt(273/M_PI)*x*y*z*rpair(sqrt(5), sqrt(11)*z);
			d1y = 3/16.0*sqrt(273/M_PI)*z*(rpair(sqrt(5/11.0), z)*(11*rpair(2*y, z) + 5) - 20/33.0);
			d1z = 33/16.0*sqrt(273/M_PI)*y*rpair(sqrt(15 - 2*sqrt(15))/sqrt(33), z)*rpair(sqrt(15 + 2*sqrt(15))/sqrt(33), z);
			
			d2xx = 3/4.0*sqrt(273)*y*z*(33*x2 + 11*y2 - 6)/sqrt(pi); //TODO
			d2yy = sqrt(273)*y*z*(24.75*x2 + 41.25*y2 - 27/2.0)/sqrt(pi); //TODO
			d2xy = sqrt(273)*x*z*(8.25*x2 + 24.75*y2 - 9/2.0)/sqrt(pi); //TODO
			d2xz = 3/4.0*sqrt(273)*x*y*(11*x2 + 11*y2 - 6)/sqrt(pi); //TODO
			d2yz = sqrt(273)*(2.0625*x4 + 12.375*x2*y2 - 9/4.0*x2 + 10.3125*y4 - 6.75*y2 + 1/2.0)/sqrt(pi); //TODO
			break;
		case 0:
			{
				double pol = -693/16.0*sqrt(13/M_PI)*rpair(sqrt(15 - 2*sqrt(15))/sqrt(33), z)*rpair(sqrt(15 + 2*sqrt(15))/sqrt(33), z);
				d1x = pol*x;
				d1y = pol*y;
			}
			d2xx = 21/16.0*sqrt(13/M_PI)*(-165*x4 - 198*x2*y2 + 108*x2 - 33*y4 + 36*y2 - 8); //TODO
			d2yy = 21/16.0*sqrt(13/M_PI)*(-33*x4 - 198*x2*y2 + 36*x2 - 165*y4 + 108*y2 - 8); //TODO
			d2xy = -63/4.0*sqrt(13/M_PI)*x*y*rpair(sqrt(5), sqrt(11)*z);			
			break;

		default: break;
		}
		break;
	case /* l = */ 7:
    switch(mm){
		case -7:
			d1x = 21.0/32.0*sqrt(715/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
			d1y = 21.0/64.0*sqrt(715/M_PI)*rpair(x, y)*rpair((sqrt(3) - 2)*x, y)*rpair((sqrt(3) + 2)*x, y);
			if(calc_hessian) {
				d2xx = 63/32.0*sqrt(715/M_PI)*y*rpair(sqrt(5 + 2*sqrt(5))*x, y)*rpair(sqrt(5 - 2*sqrt(5))*x, y);
				d2yy = -d2xx;
				d2xy = 63/32.0*sqrt(715/M_PI)*x*rpair(sqrt(5 + 2*sqrt(5))*y, x)*rpair(sqrt(5 - 2*sqrt(5))*y, x);
			}
			break;
		case -6:
			d2xz =  9/32.0*sqrt(10010/M_PI)*y*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
			d2yz =  9/32.0*sqrt(10010/M_PI)*x*rpair(sqrt(5 - 2*sqrt(5))*y, x)*rpair(sqrt(5 + 2*sqrt(5))*y, x);
			d1x  =  d2xz*z;
			d1y  =  d2yz*z; 
			d1z  =  3/32.0*sqrt(10010/M_PI)*x*y*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
			if(calc_hessian) {
				d2xx =  45/8.0*sqrt(10010/M_PI)*x*y*z*rpair(x, y);
				d2yy = -d2xx;
				d2xy =  45/32.0*sqrt(10010/M_PI)*z*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
			}
			break;
		case -5:
			d1x = -12.0/128.0*sqrt(385/M_PI)*x*y*(195*x4 - 130*x2*y2 - 120*x2 - 117*y4 + 120*y2); //TODO
			d1y = -3.0/64.0*sqrt(385/M_PI)*(65*x6 - 195*x4*y2 - 60*x4 - 585*x2*y4 + 360*x2*y2 + 91*y6 - 60*y4); //TODO

			//  f    = 0.0078125*sqrt(385)*y*(-390.0*x6 + 390.0*x4*y2 + 360.0*x4 + 702.0*x2*y4 - 720.0*x2*y2 - 78.0*y6 + 72.0*y4)/sqrt(pi); //TODO
			d2xx = 0.0078125*sqrt(385)*y*(-11700.0*x4 + 4680.0*x2*y2 + 4320.0*x2 + 1404.0*y4 - 1440.0*y2)/sqrt(pi); //TODO
			d2yy = sqrt(385)*y*(18.28125*x4 + 109.6875*x2*y2 - 33.75*x2 - 25.59375*y4 + 11.25*y2)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(385)*x*(-18.28125*x4 + 36.5625*x2*y2 + 11.25*x2 + 54.84375*y4 - 33.75*y2)/sqrt(pi); //TODO
			break;
		case -4:
			d1x  = -39/8.0*sqrt(385/M_PI)*y*z*(5*x2*rpair(x, sqrt(6/13.0)) - y2*rpair(y, sqrt(10/13.0)));
			d1y  =  39/8.0*sqrt(385/M_PI)*x*z*(5*y2*rpair(y, sqrt(6/13.0)) - x2*rpair(x, sqrt(10/13.0)));
			d1z  =  3/8.0*sqrt(385/M_PI)*x*y*rpair(x, y)*rpair(sqrt(13)*z, sqrt(3));
			d2xx = -15/2.0*sqrt(385/M_PI)*x*y*z*rpair(sqrt(13)*x, sqrt(3));
			d2yy =  15/2.0*sqrt(385/M_PI)*x*y*z*rpair(sqrt(13)*y, sqrt(3));
			d2xy = 1.0*sqrt(385)*z*(-24.375*x4 + 11.25*x2 + 24.375*y4 - 11.25*y2)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(385)*y*(-24.375*x4 + 11.25*x2 + 4.875*y4 - 3.75*y2)/sqrt(pi); //TODO
			d2yz = 1.0*sqrt(385)*x*(-4.875*x4 + 3.75*x2 + 24.375*y4 - 11.25*y2)/sqrt(pi); //TODO
			break;
		case -3:
			//  f    = 0.0078125*sqrt(35)*y*(2574.0*x6 + 4290.0*x4*y2 - 3960.0*x4 + 858.0*x2*y4 - 2640.0*x2*y2 + 1440.0*x2 - 858.0*y6 + 1320.0*y4 - 480.0*y2)/sqrt(pi); //TODO
			d1x = 12.0/128.0*sqrt(35/M_PI)*x*y*(1287*x4 + 1430*x2*y2 - 1320*x2 + 143*y4 - 440*y2 + 240); //TODO
			d1y = 3/64.0*sqrt(35/M_PI)*(429*x6 + 2145*x4*y2 - 660*x4 + 715*x2*y4 - 1320*x2*y2 + 240*x2 - 1001*y6 + 1100*y4 - 240*y2); //TODO
			
			d2xx = 0.0078125*sqrt(35)*y*(77220.0*x4 + 51480.0*x2*y2 - 47520.0*x2 + 1716.0*y4 - 5280.0*y2 + 2880.0)/sqrt(pi); //TODO
			d2yy = sqrt(35)*y*(201.09375*x4 + 134.0625*x2*y2 - 123.75*x2 - 281.53125*y4 + 206.25*y2 - 22.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(35)*x*(120.65625*x4 + 402.1875*x2*y2 - 123.75*x2 + 67.03125*y4 - 123.75*y2 + 22.5)/sqrt(pi); //TODO
			break;
		case -2:
			//  f    = 0.0078125*sqrt(70)*x*y*z*(1716.0*x4 + 3432.0*x2*y2 - 2112.0*x2 + 1716.0*y4 - 2112.0*y2 + 576.0)/sqrt(pi); //TODO
			d1x = 3/32.0*sqrt(70/M_PI)*y*z*(715*x4 + 858*x2*y2 - 528*x2 + 143*y4 - 176*y2 + 48); //TODO
			d1y = 3/32.0*sqrt(70/M_PI)*x*z*(715*y4 + 858*y2*x2 - 528*y2 + 143*x4 - 176*x2 + 48); //TODO
			d1z = 429/32.0*sqrt(70/M_PI)*x*y*rpair(sqrt(55 - 4*sqrt(55))/sqrt(143), z)*rpair(sqrt(55 + 4*sqrt(55))/sqrt(143), z);
			
			d2xx = sqrt(70)*x*y*z*(268.125*x2 + 160.875*y2 - 99.0)/sqrt(pi); //TODO
			d2yy = sqrt(70)*x*y*z*(160.875*x2 + 268.125*y2 - 99.0)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(70)*z*(67.03125*x4 + 241.3125*x2*y2 - 49.5*x2 + 67.03125*y4 - 49.5*y2 + 9/2.0)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(70)*y*(67.03125*x4 + 80.4375*x2*y2 - 49.5*x2 + 13.40625*y4 - 16.5*y2 + 9/2.0)/sqrt(pi); //TODO
			d2yz = 1.0*sqrt(70)*x*(13.40625*x4 + 80.4375*x2*y2 - 16.5*x2 + 67.03125*y4 - 49.5*y2 + 9/2.0)/sqrt(pi); //TODO
			break;
		case -1:
			//  f    = 1.0*sqrt(105)*y*(-6.703125*x6 - 20.109375*x4*y2 + 12.375*x4 - 20.109375*x2*y4 + 24.75*x2*y2 - 6.75*x2 - 6.703125*y6 + 12.375*y4 - 6.75*y2 + 1.0)/sqrt(pi); //TODO
			d1x = -1287/32.0*sqrt(105/M_PI)*x*y*rpair(sqrt(55 + 2*sqrt(220))/sqrt(143), z)*rpair(sqrt(55 - 2*sqrt(220))/sqrt(143), z);			
			d1y = -1/64.0*sqrt(105/M_PI)*(429*x6 + 3861*x4*y2 - 792*x4 + 6435*x2*y4 - 4752*x2*y2 + 432*x2 + 3003*y6 - 3960*y4 + 1296*y2 - 64); //TODO

			d2xx = 1.0*sqrt(105)*y*(-201.09375*x4 - 241.3125*x2*y2 + 148.5*x2 - 40.21875*y4 + 49.5*y2 - 27/2.0)/sqrt(pi); //TODO
			d2yy = -sqrt(105)*y*(120.65625*x4 + 402.1875*x2*y2 - 148.5*x2 + 281.53125*y4 - 247.5*y2 + 40.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(105)*x*(-40.21875*x4 - 241.3125*x2*y2 + 49.5*x2 - 201.09375*y4 + 148.5*y2 - 27/2.0)/sqrt(pi); //TODO
			break;
		case 0:
			{
				double pol = -143*9/16.0*sqrt(15/M_PI)*z*rpair(sqrt(77 - 2*sqrt(231))/sqrt(143), z)*rpair(sqrt(77 + 2*sqrt(231))/sqrt(143), z);
				d1x = pol*x;
				d1y = pol*y;

				// Analytic: d1z = 1/32.0*sqrt(15/M_PI)*(429*z2*z2*z2 - 693*z2*z2 + 315*z2 - 35);
				double const r0 = 0.40584515137739716690660641207696146;
				double const r1 = 0.74153118559939443986386477328078841;
				double const r2 = 0.94910791234275852452618968404785126;

				d1z = -429/32.0*sqrt(15/M_PI)*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
			}
			//  f    = sqrt(15)*z*(-429*x6 - 1287*x4*y2 + 594*x4 - 1287*x2*y4 + 1188*x2*y2 - 216*x2 - 429*y6 + 594*y4 - 216*y2 + 16)/(32*sqrt(pi)); //TODO
			d2xx = 9*sqrt(15)*z*(-715*x4 - 858*x2*y2 + 396*x2 - 143*y4 + 132*y2 - 24)/(16*sqrt(pi)); //TODO
			d2yy = 9*sqrt(15)*z*(-143*x4 - 858*x2*y2 + 132*x2 - 715*y4 + 396*y2 - 24)/(16*sqrt(pi)); //TODO
			d2xy = 99*sqrt(15)*x*y*z*(-13*x2 - 13*y2 + 6)/(4*sqrt(pi)); //TODO
			d2xz = 9*sqrt(15)*x*(-143*x4 - 286*x2*y2 + 132*x2 - 143*y4 + 132*y2 - 24)/(16*sqrt(pi)); //TODO
			d2yz = 9*sqrt(15)*y*(-143*x4 - 286*x2*y2 + 132*x2 - 143*y4 + 132*y2 - 24)/(16*sqrt(pi)); //TODO
			break;

		default: break;
		}
		break;
  case /* l = */ 8:
    switch(mm){
		case -8:
			//  f    = 0.001953125*sqrt(12155)*x*y*(48.0*x6 - 336.0*x4*y2 + 336.0*x2*y4 - 48.0*y6)/sqrt(pi); //TODO
			{
				// Analytic:  d1x =  3.0/32.0*sqrt(12155/M_PI)*y*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
				//            d1y = -3.0/32.0*sqrt(12155/M_PI)*x*(7*y6 - 35*y4*x2 + 21*y2*x4 - x6);
				double const r0 = 0.481574618807528644332162353056970575219;
				double const r1 = 1.253960337662703837570910978336464443221;
				double const r2 = 4.3812862675348230724046890850326954441502;
				
				d1x =  3.0/32.0*sqrt(12155/M_PI)*y*rpair(r0*x, y)*rpair(r1*x, y)*rpair(r2*x, y);
				d1y = -3.0/32.0*sqrt(12155/M_PI)*x*rpair(r0*y, x)*rpair(r1*y, x)*rpair(r2*y, x);
			}
			if(calc_hessian) {
				d2xx = 21/16.0*sqrt(12155/M_PI)*x*y*rpair(sqrt(3)*x, y)*rpair(x, sqrt(3)*y);			
				d2yy = -d2xx;
				d2xy = 21/32.0*sqrt(12155/M_PI)*rpair(x, y)*rpair((sqrt(3) + 2)*x, y)*rpair((sqrt(3) - 2)*x, y);
			}
			break;
		case -7:
			{
				d1x = 21/32.0*sqrt(12155/M_PI)*x*y*z*rpair(x, sqrt(3)*y)*rpair(sqrt(3)*x, y);
				d1y = 21/64.0*sqrt(12155/M_PI)*z*rpair(x, y)*rpair((sqrt(3) + 2)*x, y)*rpair((sqrt(3) - 2)*x, y);
				
				// Analytic: d1z = 3.0/64.0*sqrt(12155/M_PI)*y*(7.0*x6 - 35.0*x4*y2 + 21.0*x2*y4 - y6);
				double const r0 = 0.481574618807528644332162353056970575219;
				double const r1 = 1.253960337662703837570910978336464443221;
				double const r2 = 4.3812862675348230724046890850326954441502;
				
				d1z = 3.0/64.0*sqrt(12155/M_PI)*y*rpair(r0*x, y)*rpair(r1*x, y)*rpair(r2*x, y);
			}
			//  f    = 0.0078125*sqrt(12155)*y*z*(42.0*x6 - 210.0*x4*y2 + 126.0*x2*y4 - 6.0*y6)/sqrt(pi); //TODO
			d2xx = 0.0078125*sqrt(12155)*y*z*(1260.0*x4 - 2520.0*x2*y2 + 252.0*y4)/sqrt(pi); //TODO
			d2yy = -sqrt(12155)*y*z*(9.84375*x4 - 19.6875*x2*y2 + 1.96875*y4)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(12155)*x*z*(1.96875*x4 - 19.6875*x2*y2 + 9.84375*y4)/sqrt(pi); //TODO
			d2xz = 0.0078125*sqrt(12155)*x*y*(252.0*x4 - 840.0*x2*y2 + 252.0*y4)/sqrt(pi); //TODO
			d2yz = sqrt(12155)*(0.328125*x6 - 4.921875*x4*y2 + 4.921875*x2*y4 - 0.328125*y6)/sqrt(pi); //TODO
			break;
		case -6:
			//  f    = 0.00390625*sqrt(14586)*x*y*(-180.0*x6 + 420.0*x4*y2 + 168.0*x4 + 420.0*x2*y4 - 560.0*x2*y2 - 180.0*y6 + 168.0*y4)/sqrt(pi); //TODO
			d1x = -3/64.0*sqrt(14586/M_PI)*y*(105*x6 - 175*x4*y2 - 70*x4 - 105*x2*y4 + 140*x2*y2 + 15*y6 - 14*y4); //TODO
			d1y = -3/64.0*sqrt(14586/M_PI)*x*(105*y6 - 175*y4*x2 - 70*y4 - 105*y2*x4 + 140*y2*x2 + 15*x6 - 14*x4); //TODO

			d2xx = sqrt(14586)*x*y*(-29.53125*x4 + 32.8125*x2*y2 + 13.125*x2 + 9.84375*y4 - 13.125*y2)/sqrt(pi); //TODO
			d2yy = sqrt(14586)*x*y*(9.84375*x4 + 32.8125*x2*y2 - 13.125*x2 - 29.53125*y4 + 13.125*y2)/sqrt(pi); //TODO
			d2xy = sqrt(14586)*(-4.921875*x6 + 24.609375*x4*y2 + 3.28125*x4 + 24.609375*x2*y4 - 19.6875*x2*y2 - 4.921875*y6 + 3.28125*y4)/sqrt(pi); //TODO
			break;
		case -5:
			//  f    = 0.0078125*sqrt(17017)*y*z*(-150.0*x6 + 150.0*x4*y2 + 120.0*x4 + 270.0*x2*y4 - 240.0*x2*y2 - 30.0*y6 + 24.0*y4)/sqrt(pi); //TODO
			d1x = -1/32.0*sqrt(17017/M_PI)*x*y*z*(rpair(5*rpair(sqrt(3)*x, y) - 4, sqrt(10)*rpair(2*y, 1))- 6);
			d1y = -15/64.0*sqrt(17017/M_PI)*z*(5*x6 - 15*x4*y2 - 4*x4 - 45*x2*y4 + 24*x2*y2 + 7*y6 - 4*y4); //TODO
			d1z = -3.0/64.0*sqrt(17017/M_PI)*y*rpair(1, sqrt(5)*z)*rpair(sqrt(5 - 2*sqrt(5))*x, y)*rpair(sqrt(5 + 2*sqrt(5))*x, y);
			
			d2xx = 0.0078125*sqrt(17017)*y*z*(-4500.0*x4 + 1800.0*x2*y2 + 1440.0*x2 + 540.0*y4 - 480.0*y2)/sqrt(pi); //TODO
			d2yy = sqrt(17017)*y*z*(7.03125*x4 + 42.1875*x2*y2 - 11.25*x2 - 9.84375*y4 + 3.75*y2)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(17017)*x*z*(-7.03125*x4 + 14.0625*x2*y2 + 3.75*x2 + 21.09375*y4 - 11.25*y2)/sqrt(pi); //TODO
			d2xz = 0.0078125*sqrt(17017)*x*y*(-900.0*x4 + 600.0*x2*y2 + 480.0*x2 + 540.0*y4 - 480.0*y2)/sqrt(pi); //TODO
			d2yz = sqrt(17017)*(-1.171875*x6 + 3.515625*x4*y2 + 0.9375*x4 + 10.546875*x2*y4 - 45/8.0*x2*y2 - 1.640625*y6 + 0.9375*y4)/sqrt(pi); //TODO
			break;
		case -4:
			//  f    = 0.00390625*sqrt(1309)*x*y*(1560.0*x6 + 1560.0*x4*y2 - 2496.0*x4 - 1560.0*x2*y4 + 960.0*x2 - 1560.0*y6 + 2496.0*y4 - 960.0*y2)/sqrt(pi); //TODO
			d1x =  3/32.0*sqrt(1309/M_PI)*y*(120*x2 - 520*x4 + 455*x6 - 40*y2 + 325*x4*y2 + 104*y4 - 195*x2*y4 - 65*y6); //TODO
			d1y = -3/32.0*sqrt(1309/M_PI)*x*(120*y2 - 520*y4 + 455*y6 - 40*x2 + 325*y4*x2 + 104*x4 - 195*y2*x4 - 65*x6); //TODO
			
			d2xx = sqrt(1309)*x*y*(255.9375*x4 + 121.875*x2*y2 - 195.0*x2 - 36.5625*y4 + 22.5)/sqrt(pi); //TODO
			d2yy = -sqrt(1309)*x*y*(-36.5625*x4 + 121.875*x2*y2 + 255.9375*y4 - 195.0*y2 + 22.5)/sqrt(pi); //TODO
			d2xy = sqrt(1309)*(42.65625*x6 + 91.40625*x4*y2 - 48.75*x4 - 91.40625*x2*y4 + 11.25*x2 - 42.65625*y6 + 48.75*y4 - 11.25*y2)/sqrt(pi); //TODO
			break;
		case -3:
			//  f    = 0.0078125*sqrt(19635)*y*z*(234.0*x6 + 390.0*x4*y2 - 312.0*x4 + 78.0*x2*y4 - 208.0*x2*y2 + 96.0*x2 - 78.0*y6 + 104.0*y4 - 32.0*y2)/sqrt(pi); //TODO
			d1x = 1/32.0*sqrt(19635/M_PI)*x*y*z*(351*x4 + 390*x2*y2 - 312*x2 + 39*y4 - 104*y2 + 48); //TODO
			d1y = 1/64.0*sqrt(19635/M_PI)*z*(48*x2 - 156*x4 + 117*x6 - 48*y2 - 312*x2*y2 + 585*x4*y2 + 260*y4 + 195*x2*y4 - 273*y6); //TODO
			d1z = 39/64.0*sqrt(19635/M_PI)*y*rpair(sqrt(3)*x, y)*rpair(sqrt(13 - 2*sqrt(13))/sqrt(39), z)*rpair(sqrt(13 + 2*sqrt(13))/sqrt(39), z);
				
			d2xx = 0.0078125*sqrt(19635)*y*z*(7020.0*x4 + 4680.0*x2*y2 - 3744.0*x2 + 156.0*y4 - 416.0*y2 + 192.0)/sqrt(pi); //TODO
			d2yy = sqrt(19635)*y*z*(18.28125*x4 + 12.1875*x2*y2 - 9.75*x2 - 25.59375*y4 + 16.25*y2 - 1.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(19635)*x*z*(10.96875*x4 + 36.5625*x2*y2 - 9.75*x2 + 6.09375*y4 - 9.75*y2 + 1.5)/sqrt(pi); //TODO
			d2xz = 0.0078125*sqrt(19635)*x*y*(1404.0*x4 + 1560.0*x2*y2 - 1248.0*x2 + 156.0*y4 - 416.0*y2 + 192.0)/sqrt(pi); //TODO
			d2yz = sqrt(19635)*(1.828125*x6 + 9.140625*x4*y2 - 2.4375*x4 + 3.046875*x2*y4 - 4.875*x2*y2 + 3/4.0*x2 - 4.265625*y6 + 4.0625*y4 - 3/4.0*y2)/sqrt(pi); //TODO
			break;
		case -2:
			//  f    = 1.0*sqrt(1190)*x*y*(-6.703125*x6 - 20.109375*x4*y2 + 13.40625*x4 - 20.109375*x2*y4 + 26.8125*x2*y2 - 8.25*x2 - 6.703125*y6 + 13.40625*y4 - 8.25*y2 + 1.5)/sqrt(pi); //TODO
			d1x = -3/64.0*sqrt(1190/M_PI)*y*(1001*x6 + 2145*x4*y2 - 1430*x4 + 1287*x2*y4 - 1716*x2*y2 + 528*x2 + 143*y6 - 286*y4 + 176*y2 - 32); //TODO
			d1y = -3/64.0*sqrt(1190/M_PI)*x*(1001*y6 + 2145*y4*x2 - 1430*y4 + 1287*y2*x4 - 1716*y2*x2 + 528*y2 + 143*x6 - 286*x4 + 176*x2 - 32); //TODO
			
			d2xx = -sqrt(1190)*x*y*(281.53125*x4 + 402.1875*x2*y2 - 268.125*x2 + 120.65625*y4 - 160.875*y2 + 49.5)/sqrt(pi); //TODO
			d2yy = -sqrt(1190)*x*y*(120.65625*x4 + 402.1875*x2*y2 - 160.875*x2 + 281.53125*y4 - 268.125*y2 + 49.5)/sqrt(pi); //TODO
			d2xy = sqrt(1190)*(-46.921875*x6 - 301.640625*x4*y2 + 67.03125*x4 - 301.640625*x2*y4 + 241.3125*x2*y2 - 24.75*x2 - 46.921875*y6 + 67.03125*y4 - 24.75*y2 + 1.5)/sqrt(pi); //TODO
			break;
		case -1:
			//  f    = 1.0*sqrt(17)*y*z*(-33.515625*x6 - 100.546875*x4*y2 + 53.625*x4 - 100.546875*x2*y4 + 107.25*x2*y2 - 24.75*x2 - 33.515625*y6 + 53.625*y4 - 24.75*y2 + 3.0)/sqrt(pi); //TODO
			{
				d1x = -6435/32.0*sqrt(17/M_PI)*x*y*z*rpair(sqrt(91 + 4*sqrt(91))/sqrt(195), z)*rpair(sqrt(91 - 4*sqrt(91))/sqrt(195), z);
				d1y = -3/64.0*sqrt(17/M_PI)*z*(715*x6 + 6435*x4*y2 - 1144*x4 + 10725*x2*y4 - 6864*x2*y2 + 528*x2 + 5005*y6 - 5720*y4 + 1584*y2 - 64); //TODO
				
				// Analityc: d1z = 3/64.0*sqrt(17/M_PI)*y*(715*z6 - 1001*z4 + 385*z2 - 35);
				double const r0 = 0.36311746382617815871075206870865921;
				double const r1 = 0.67718627951073775344588542709134245;
				double const r2 = 0.8997579954114601573123452444183380;
				
				d1z = -2145/64.0*sqrt(17/M_PI)*y*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
			}
			d2xx = 1.0*sqrt(17)*y*z*(-1005.46875*x4 - 1206.5625*x2*y2 + 643.5*x2 - 201.09375*y4 + 214.5*y2 - 49.5)/sqrt(pi); //TODO
			d2yy = -sqrt(17)*y*z*(603.28125*x4 + 2010.9375*x2*y2 - 643.5*x2 + 1407.65625*y4 - 1072.5*y2 + 148.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(17)*x*z*(-201.09375*x4 - 1206.5625*x2*y2 + 214.5*x2 - 1005.46875*y4 + 643.5*y2 - 49.5)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(17)*x*y*(-201.09375*x4 - 402.1875*x2*y2 + 214.5*x2 - 201.09375*y4 + 214.5*y2 - 49.5)/sqrt(pi); //TODO
			d2yz = sqrt(17)*(-33.515625*x6 - 301.640625*x4*y2 + 53.625*x4 - 502.734375*x2*y4 + 321.75*x2*y2 - 24.75*x2 - 234.609375*y6 + 268.125*y4 - 74.25*y2 + 3.0)/sqrt(pi); //TODO
			break;
		case 0:
			{
				// Analytic: double pol = -9/32.0*sqrt(17/M_PI)*(715*z6 - 1001*z4 + 385*z2 - 35);
				double const r0 = 0.36311746382617815871075206870865921;
				double const r1 = 0.67718627951073775344588542709134245;
				double const r2 = 0.89975799541146015731234524441833796;
				
				double pol = 6435/32.0*sqrt(17/M_PI)*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
				d1x = pol*x;
				d1y = pol*y;
			}
			d2xx = 9*sqrt(17)*(5005*x6 + 10725*x4*y2 - 5720*x4 + 6435*x2*y4 - 6864*x2*y2 + 1584*x2 + 715*y6 - 1144*y4 + 528*y2 - 64)/(32*sqrt(pi)); //TODO
			d2yy = 9*sqrt(17)*(715*x6 + 6435*x4*y2 - 1144*x4 + 10725*x2*y4 - 6864*x2*y2 + 528*x2 + 5005*y6 - 5720*y4 + 1584*y2 - 64)/(32*sqrt(pi)); //TODO
			//			d2xy = 99/16.0*sqrt(17/M_PI)*x*y*(195*z2*z2 - 182*z2 + 35);
			d2xy = 195*99/16.0*sqrt(17/M_PI)*x*y*rpair(sqrt((7 - 4*sqrt(7/13.0))/15.0), z)*rpair(sqrt((7 + 4*sqrt(7/13.0))/15.0), z);
			break;

		default: break;
		}
		break;
  case /* l = */ 9:
    switch(mm){
		case -9:
			//  f    = 1.0*sqrt(461890)*y*(0.017578125*x8 - 0.1640625*x6*y2 + 0.24609375*x4*y4 - 0.0703125*x2*y6 + 0.001953125*y8)/sqrt(pi); //TODO
			d1x = 9/64.0*sqrt(461890/M_PI)*x*y*rpair(x, y)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
			d1y = 9/512.0*sqrt(461890/M_PI)*
				rpair((1 - sqrt(2) - sqrt(4 - 2*sqrt(2)))*x, y)*rpair((1 - sqrt(2) + sqrt(4 - 2*sqrt(2)))*x, y)*
				rpair((1 + sqrt(2) + sqrt(4 + 2*sqrt(2)))*x, y)*rpair((1 + sqrt(2) - sqrt(4 + 2*sqrt(2)))*x, y);
			
			d2xx = 1.0*sqrt(461890)*y*(0.984375*x6 - 4.921875*x4*y2 + 2.953125*x2*y4 - 0.140625*y6)/sqrt(pi); //TODO
			d2yy = -sqrt(461890)*y*(0.984375*x6 - 4.921875*x4*y2 + 2.953125*x2*y4 - 0.140625*y6)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(461890)*x*(0.140625*x6 - 2.953125*x4*y2 + 4.921875*x2*y4 - 0.984375*y6)/sqrt(pi); //TODO
			break;
		case -8:
			//  f    = 0.001953125*sqrt(230945)*x*y*z*(48.0*x6 - 336.0*x4*y2 + 336.0*x2*y4 - 48.0*y6)/sqrt(pi); //TODO
			{
				// Analytic: d1x = 3/32.0*sqrt(230945/M_PI)*y*z*(7*x6 - 35*x4*y2 + 21*x2*y4 - y6);
				//           d1y = 3/32.0*sqrt(230945/M_PI)*x*z*(7*y6 - 35*y4*x2 + 21*y2*x4 - x6);
				double const r0 = 0.481574618807528644332162353056970575219;
				double const r1 = 1.253960337662703837570910978336464443221;
				double const r2 = 4.3812862675348230724046890850326954441502;

				d1x =  3/32.0*sqrt(230945/M_PI)*y*z*rpair(r0*x, y)*rpair(r1*x, y)*rpair(r2*x, y);
				d1y = -3/32.0*sqrt(230945/M_PI)*x*z*rpair(r0*y, x)*rpair(r1*y, x)*rpair(r2*y, x);

				d1z = 3/32.0*sqrt(230945/M_PI)*x*y*rpair(x, y)*rpair((sqrt(2) + 1)*x, y)*rpair((sqrt(2) - 1)*x, y);
			}
			d2xx = sqrt(230945)*x*y*z*(3.9375*x4 - 13.125*x2*y2 + 3.9375*y4)/sqrt(pi); //TODO
			d2yy = -sqrt(230945)*x*y*z*(3.9375*x4 - 13.125*x2*y2 + 3.9375*y4)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(230945)*z*(0.65625*x6 - 9.84375*x4*y2 + 9.84375*x2*y4 - 0.65625*y6)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(230945)*y*(0.65625*x6 - 3.28125*x4*y2 + 1.96875*x2*y4 - 0.09375*y6)/sqrt(pi); //TODO
			d2yz = 1.0*sqrt(230945)*x*(0.09375*x6 - 1.96875*x4*y2 + 3.28125*x2*y4 - 0.65625*y6)/sqrt(pi); //TODO
			break;
		case -7:
			//  f    = 0.0009765625*sqrt(27170)*y*(-714.0*x8 + 2856.0*x6*y2 + 672.0*x6 + 1428.0*x4*y4 - 3360.0*x4*y2 - 2040.0*x2*y6 + 2016.0*x2*y4 + 102.0*y8 - 96.0*y6)/sqrt(pi); //TODO
			d1x = -3/64.0*sqrt(27170/M_PI)*x*y*(119*x6 - 357*x4*y2 - 84*x4 - 119*x2*y4 + 280*x2*y2 + 85*y6 - 84*y4); //TODO
			d1y = -3/512.0*sqrt(27170/M_PI)*(119*x8 - 1428*x6*y2 - 112*x6 - 1190*x4*y4 + 1680*x4*y2 + 2380*x2*y6 - 1680*x2*y4 - 153*y8 + 112*y6);  //TODO
			
			d2xx = 0.0009765625*sqrt(27170)*y*(-39984.0*x6 + 85680.0*x4*y2 + 20160.0*x4 + 17136.0*x2*y4 - 40320.0*x2*y2 - 4080.0*y6 + 4032.0*y4)/sqrt(pi); //TODO
			d2yy = sqrt(27170)*y*(16.734375*x6 + 27.890625*x4*y2 - 19.6875*x4 - 83.671875*x2*y4 + 39.375*x2*y2 + 7.171875*y6 - 3.9375*y4)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(27170)*x*(-5.578125*x6 + 50.203125*x4*y2 + 3.9375*x4 + 27.890625*x2*y4 - 39.375*x2*y2 - 27.890625*y6 + 19.6875*y4)/sqrt(pi); //TODO
			break;
		case -6:
			d1x = 1/64.0*sqrt(81510/M_PI)*3*y*z*(17*z2 - 3)*(5*x4 - 10*x2*y2 + y4);
			d1y = 1/64.0*sqrt(81510/M_PI)*3*x*z*(17*z2 - 3)*(x4 - 10*x2*y2 + 5*y4);
			d1z = 1/64.0*sqrt(81510/M_PI)*3*x*y*(x2 - 3*y2)*(3*x2 - y2)*(17*z2 - 1);
			
			d2xx = 17.0/64.0*sqrt(81510/M_PI)*60*x*y*z*rpair(x, y)*rpair(z, sqrt(3/17.0));
			d2yy = -d2xx;
			d2zz = 1.0/64.0*sqrt(81510/M_PI)*102*x*y*z*(x2 - 3*y2)*(3*x2 - y2);
			d2xy = 1.0/64.0*sqrt(81510/M_PI)*15*z*(17*z2 - 3)*(x4 - 6*x2*y2 + y4);
			d2xz = 1.0/64.0*sqrt(81510/M_PI)*9*y*(5*x4 - 10*x2*y2 + y4)*(17*z2 - 1);
			d2yz = 1.0/64.0*sqrt(81510/M_PI)*9*x*(x4 - 10*x2*y2 + 5*y4)*(17*z2 - 1);
			break;
		case -5:
			//  f    = 0.001953125*sqrt(5434)*y*(2550.0*x8 - 4200.0*x6 - 7140.0*x4*y4 + 4200.0*x4*y2 + 1680.0*x4 - 4080.0*x2*y6 + 7560.0*x2*y4 - 3360.0*x2*y2 + 510.0*y8 - 840.0*y6 + 336.0*y4)/sqrt(pi); //TODO
			d1x = 15/32.0*sqrt(5434/M_PI)*x*y*(85*x6 - 105*x4 - 119*x2*y4 + 70*x2*y2 + 28*x2 - 34*y6 + 63*y4 - 28*y2); //TODO
			d1y = 15/256.0*sqrt(5434/M_PI)*(85*x8 - 140*x6 - 1190*x4*y4 + 420*x4*y2 + 56*x4 - 952*x2*y6 + 1260*x2*y4 - 336*x2*y2 + 153*y8 - 196*y6 + 56*y4); //TODO
			
			d2xx = 0.001953125*sqrt(5434)*y*(142800.0*x6 - 126000.0*x4 - 85680.0*x2*y4 + 50400.0*x2*y2 + 20160.0*x2 - 8160.0*y6 + 15120.0*y4 - 6720.0*y2)/sqrt(pi); //TODO
			d2yy = -sqrt(5434)*y*(278.90625*x4*y2 - 49.21875*x4 + 334.6875*x2*y4 - 295.3125*x2*y2 + 39.375*x2 - 71.71875*y6 + 68.90625*y4 - 13.125*y2)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(5434)*x*(39.84375*x6 - 49.21875*x4 - 278.90625*x2*y4 + 98.4375*x2*y2 + 13.125*x2 - 111.5625*y6 + 147.65625*y4 - 39.375*y2)/sqrt(pi); //TODO
			break;
		case -4:
			//  f    = 0.00390625*sqrt(95095)*x*y*z*(408.0*x6 + 408.0*x4*y2 - 576.0*x4 - 408.0*x2*y4 + 192.0*x2 - 408.0*y6 + 576.0*y4 - 192.0*y2)/sqrt(pi); //TODO
			d1x = 3/32.0*sqrt(95095/M_PI)*y*z*(119*x6 + 85*x4*y2 - 120*x4 - 51*x2*y4 + 24*x2 - 17*y6 + 24*y4 - 8*y2); //TODO
			d1y = 3/32.0*sqrt(95095/M_PI)*x*z*(17*x6 + 51*x4*y2 - 24*x4 - 85*x2*y4 + 8*x2 - 119*y6 + 120*y4 - 24*y2); //TODO
			d1z = 51/32.0*sqrt(95095/M_PI)*x*y*rpair(x, y)*rpair(sqrt(5 - 2*sqrt(2))/sqrt(17), z)*rpair(sqrt(5 + 2*sqrt(2))/sqrt(17), z);
			
			d2xx = sqrt(95095)*x*y*z*(66.9375*x4 + 31.875*x2*y2 - 45.0*x2 - 9.5625*y4 + 9/2.0)/sqrt(pi); //TODO
			d2yy = -sqrt(95095)*x*y*z*(-9.5625*x4 + 31.875*x2*y2 + 66.9375*y4 - 45.0*y2 + 9/2.0)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(95095)*z*(11.15625*x6 + 23.90625*x4*y2 - 11.25*x4 - 23.90625*x2*y4 + 9/4.0*x2 - 11.15625*y6 + 11.25*y4 - 9/4.0*y2)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(95095)*y*(11.15625*x6 + 7.96875*x4*y2 - 11.25*x4 - 4.78125*x2*y4 + 9/4.0*x2 - 1.59375*y6 + 9/4.0*y4 - 3/4.0*y2)/sqrt(pi); //TODO
			d2yz = 1.0*sqrt(95095)*x*(1.59375*x6 + 4.78125*x4*y2 - 9/4.0*x4 - 7.96875*x2*y4 + 3/4.0*x2 - 11.15625*y6 + 11.25*y4 - 9/4.0*y2)/sqrt(pi); //TODO
			break;
		case -3:
			//  f    = 1.0*sqrt(43890)*y*(-2.58984375*x8 - 6.90625*x6*y2 + 5.484375*x6 - 5.1796875*x4*y4 + 9.140625*x4*y2 - 3.65625*x4 + 1.828125*x2*y4 - 2.4375*x2*y2 + 3/4.0*x2 + 0.86328125*y8 - 1.828125*y6 + 1.21875*y4 - 0.25*y2)/sqrt(pi); //TODO
			d1x = -3/32.0*sqrt(43890/M_PI)*x*y*(221*x6 + 442*x4*y2 - 351*x4 + 221*x2*y4 - 390*x2*y2 + 156*x2 - 39*y4 + 52*y2 - 16); //TODO
			d1y = -3/256.0*sqrt(43890/M_PI)*(221*x8 + 1768*x6*y2 - 468*x6 + 2210*x4*y4 - 2340*x4*y2 + 312*x4 - 780*x2*y4 + 624*x2*y2 - 64*x2 - 663*y8 + 1092*y6 - 520*y4 + 64*y2); //TODO	
			
			d2xx = 1.0*sqrt(43890)*y*(-145.03125*x6 - 207.1875*x4*y2 + 164.53125*x4 - 62.15625*x2*y4 + 109.6875*x2*y2 - 43.875*x2 + 3.65625*y4 - 4.875*y2 + 1.5)/sqrt(pi); //TODO
			d2yy = -sqrt(43890)*y*(41.4375*x6 + 103.59375*x4*y2 - 54.84375*x4 - 36.5625*x2*y2 + 14.625*x2 - 62.15625*y6 + 76.78125*y4 - 24.375*y2 + 1.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(43890)*x*(-20.71875*x6 - 124.3125*x4*y2 + 32.90625*x4 - 103.59375*x2*y4 + 109.6875*x2*y2 - 14.625*x2 + 18.28125*y4 - 14.625*y2 + 1.5)/sqrt(pi); //TODO
			break;
		case -2:
			//  f    = 1.0*sqrt(2090)*x*y*z*(-10.359375*x6 - 31.078125*x4*y2 + 18.28125*x4 - 31.078125*x2*y4 + 36.5625*x2*y2 - 9.75*x2 - 10.359375*y6 + 18.28125*y4 - 9.75*y2 + 1.5)/sqrt(pi); //TODO
			{
				d1x = -3/64.0*sqrt(2090/M_PI)*y*z*(1547*x6 + 3315*x4*y2 - 1950*x4 + 1989*x2*y4 - 2340*x2*y2 + 624*x2 + 221*y6 - 390*y4 + 208*y2 - 32);  //TODO
				d1y = -3/64.0*sqrt(2090/M_PI)*x*z*(1547*y6 + 3315*y4*x2 - 1950*y4 + 1989*y2*x4 - 2340*y2*x2 + 624*y2 + 221*x6 - 390*x4 + 208*x2 - 32);  //TODO
				
				// Analytic: d1z = 3/64.0*sqrt(2090/M_PI)*x*y*(221*z6 - 273*z4 + 91*z2 - 7); //TODO
				double const r0 = 0.33162955360278980220085003507847223;
				double const r1 = 0.62737813976261871800246540074189546;
				double const r2 = 0.85540202628487456905775301621747783;
				
				d1z = -663/64.0*sqrt(2090/M_PI)*x*y*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
			}
			d2xx = -sqrt(2090)*x*y*z*(435.09375*x4 + 621.5625*x2*y2 - 365.625*x2 + 186.46875*y4 - 219.375*y2 + 58.5)/sqrt(pi); //TODO
			d2yy = -sqrt(2090)*x*y*z*(186.46875*x4 + 621.5625*x2*y2 - 219.375*x2 + 435.09375*y4 - 365.625*y2 + 58.5)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(2090)*z*(-72.515625*x6 - 466.171875*x4*y2 + 91.40625*x4 - 466.171875*x2*y4 + 329.0625*x2*y2 - 29.25*x2 - 72.515625*y6 + 91.40625*y4 - 29.25*y2 + 1.5)/sqrt(pi); //TODO
			d2xz = 1.0*sqrt(2090)*y*(-72.515625*x6 - 155.390625*x4*y2 + 91.40625*x4 - 93.234375*x2*y4 + 109.6875*x2*y2 - 29.25*x2 - 10.359375*y6 + 18.28125*y4 - 9.75*y2 + 1.5)/sqrt(pi); //TODO
			d2yz = 1.0*sqrt(2090)*x*(-10.359375*x6 - 93.234375*x4*y2 + 18.28125*x4 - 155.390625*x2*y4 + 109.6875*x2*y2 - 9.75*x2 - 72.515625*y6 + 91.40625*y4 - 29.25*y2 + 1.5)/sqrt(pi); //TODO
			break;
		case -1:
			//  f    = 0.01171875*sqrt(95)*y*(2431*x8 + 9724*x6*y2 - 5720*x6 + 14586*x4*y4 - 17160*x4*y2 + 4576*x4 + 9724*x2*y6 - 17160*x2*y4 + 9152*x2*y2 - 1408*x2 + 2431*y8 - 5720*y6 + 4576*y4 - 1408*y2 + 128)/sqrt(pi); //TODO
			{
				// Analytic: d1x = -33/32.0*sqrt(95/M_PI)*x*y*(221*z6 - 273*z4 + 91*z2 - 7);
				double const r0 = 0.33162955360278980220085003507847223;
				double const r1 = 0.62737813976261871800246540074189546;
				double const r2 = 0.85540202628487456905775301621747783;
				d1x = 7293/32.0*sqrt(95/M_PI)*x*y*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
				
				d1y = 3/256.0*sqrt(95/M_PI)*(2431*x8 + 29172*x6*y2 - 5720*x6 + 72930*x4*y4 - 51480*x4*y2 + 4576*x4 + 68068*x2*y6 - 85800*x2*y4 + 27456*x2*y2 - 1408*x2 + 21879*y8 - 40040*y6 + 22880*y4 - 4224*y2 + 128); //TODO
			}
			d2xx = 1.03125*sqrt(95)*y*(1547*x6 + 3315*x4*y2 - 1950*x4 + 1989*x2*y4 - 2340*x2*y2 + 624*x2 + 221*y6 - 390*y4 + 208*y2 - 32)/sqrt(pi); //TODO
			d2yy = sqrt(95)*y*(683.71875*x6 + 3418.59375*x4*y2 - 1206.5625*x4 + 4786.03125*x2*y4 - 4021.875*x2*y2 + 643.5*x2 + 2051.15625*y6 - 2815.3125*y4 + 1072.5*y2 - 99.0)/sqrt(pi); //TODO
			d2xy = 1.0*sqrt(95)*x*(227.90625*x6 + 2051.15625*x4*y2 - 402.1875*x4 + 3418.59375*x2*y4 - 2413.125*x2*y2 + 214.5*x2 + 1595.34375*y6 - 2010.9375*y4 + 643.5*y2 - 33.0)/sqrt(pi); //TODO
			break;
		case 0:
			//  f    = sqrt(19)*z*(12155*x8 + 48620*x6*y2 - 22880*x6 + 72930*x4*y4 - 68640*x4*y2 + 13728*x4 + 48620*x2*y6 - 68640*x2*y4 + 27456*x2*y2 - 2816*x2 + 12155*y8 - 22880*y6 + 13728*y4 - 2816*y2 + 128)/(256*sqrt(pi)); //TODO
			{
							// Analytic: double pol = 11/32.0*sqrt(19/M_PI)*z*((z2*(1755 - 1105*z2) - 819)*z2 + 105);
				double const r0 = 0.45908295445521662139638697796909642;
				double const r1 = 0.73174254576912532452982321723182069;
				double const r2 = 0.91762246145004323870310163513571051;
				double pol = 12155/32.0*sqrt(19/M_PI)*z*rpair(r0, z)*rpair(r1, z)*rpair(r2, z);
				d1x = pol*x;
				d1y = pol*y;
			}
			{
				// Analytic: d1z = 1/256.0*sqrt(19/M_PI)*((((12155*z2 - 25740)*z2 + 18018)*z2 - 4620)*z2 + 315);
				double const r0 = 0.32425342340380892903853801464333661;
				double const r1 = 0.61337143270059039730870203934147418;
				double const r2 = 0.83603110732663579429942978806973488;
				double const r3 = 0.96816023950762608983557620290367287;
				d1z = 12155/256.0*sqrt(19/M_PI)*rpair(r0, z)*rpair(r1, z)*rpair(r2, z)*rpair(r3, z);
			}
			d2xx = 11/32.0*sqrt(19/M_PI)*z*(7735*x6 + 16575*x4*y2 - 7800*x4 + 9945*x2*y4 - 9360*x2*y2 + 1872*x2 + 1105*y6 - 1560*y4 + 624*y2 - 64); //TODO
			d2yy = 11/32.0*sqrt(19/M_PI)*z*(1105*x6 + 9945*x4*y2 - 1560*x4 + 16575*x2*y4 - 9360*x2*y2 + 624*x2 + 7735*y6 - 7800*y4 + 1872*y2 - 64); //TODO
			d2xy = 429/16.0*sqrt(19/M_PI)*x*y*z*(85*x4 + 170*x2*y2 - 80*x2 + 85*y4 - 80*y2 + 16); //TODO
			d2xz = 11/32.0*sqrt(19/M_PI)*x*(1105*x6 + 3315*x4*y2 - 1560*x4 + 3315*x2*y4 - 3120*x2*y2 + 624*x2 + 1105*y6 - 1560*y4 + 624*y2 - 64); //TODO
			d2yz = 11/32.0*sqrt(19/M_PI)*y*(1105*x6 + 3315*x4*y2 - 1560*x4 + 3315*x2*y4 - 3120*x2*y2 + 624*x2 + 1105*y6 - 1560*y4 + 624*y2 - 64); //TODO
			break;
		default: break;
		}
		break;

	default: break;
	}
	
	double const d2yx = d2xy;
	double const d2zx = d2xz;
	double const d2zy = d2yz;
	
	double const radial_proj = x*d1x + y*d1y + z*d1z;
	
	*gx = d1x - x*radial_proj;
	*gy = d1y - y*radial_proj;
	*gz = d1z - z*radial_proj;

	if(calc_hessian) {
		double const hessian_proj_x = x*d2xx + y*d2xy + z*d2xz;
		double const hessian_proj_y = x*d2yx + y*d2yy + z*d2yz;
		double const hessian_proj_z = x*d2zx + y*d2zy + z*d2zz;
		double const radial_curv = x*hessian_proj_x + y*hessian_proj_y + z*hessian_proj_z;
		
		*hxx = d2xx - 2*x*hessian_proj_x + x2*radial_curv - 2*x*d1x + radial_proj*(3*x2 - 1);
		*hyy = d2yy - 2*y*hessian_proj_y + y2*radial_curv - 2*y*d1y + radial_proj*(3*y2 - 1);
		*hzz = d2zz - 2*z*hessian_proj_z + z2*radial_curv - 2*z*d1z + radial_proj*(3*z2 - 1);
		*hxy = d2xy - (x*hessian_proj_y + y*hessian_proj_x) + x*y*radial_curv - (x*d1y + y*d1x) + 3*x*y*radial_proj;
		*hxz = d2xz - (x*hessian_proj_z + z*hessian_proj_x) + x*z*radial_curv - (x*d1z + z*d1x) + 3*x*z*radial_proj;
		*hyz = d2yz - (y*hessian_proj_z + z*hessian_proj_y) + y*z*radial_curv - (y*d1z + z*d1y) + 3*y*z*radial_proj;
	}
	
}

///////////////////////////////////////////////////////////////////
//
// GRADIENT STUFF
//
///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(normalized_cartesian_real_rgradient)(int const ll, int const mm, double x, double y, double z, double * gx, double * gy, double * gz) {

	SH_FUNC(normalized_cartesian_real_r2hessian)(ll, mm, x, y, z, gx, gy, gz, NULL, NULL, NULL, NULL, NULL, NULL);
		
	double const radial_projection = x*(*gx) + y*(*gy) + z*(*gz);
	*gx -= x*radial_projection;
	*gy -= y*radial_projection;
	*gz -= z*radial_projection;
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(cartesian_real_rgradient)(int const ll, int const mm, double xx, double yy, double zz, double * gx, double * gy, double * gz) {

#ifdef __cplusplus
	using namespace internal;
#endif
	
	double norm = SH_FUNC(normalize)(&xx, &yy, &zz);
	
	if(norm < 1.0e-15) {
		*gx = 0.0;
		*gy = 0.0;
		*gz = 0.0;

		return;
	}
	
	SH_FUNC(normalized_cartesian_real_rgradient)(ll, mm, xx, yy, zz, gx, gy, gz);
}

///////////////////////////////////////////////////////////////////

SH_GPU_FUNCTION static void SH_FUNC(angular_real_rgradient)(int const ll, int const mm, double theta, double phi, double * gx, double * gy, double * gz) {

#ifdef __cplusplus
	using namespace internal;
#endif

	double xx, yy, zz;
	SH_FUNC(angular_to_normalized_cartesian)(theta, phi, &xx, &yy, &zz);
	SH_FUNC(normalized_cartesian_real_rgradient)(ll, mm, xx, yy, zz, gx, gy, gz);
}

///////////////////////////////////////////////////////////////////

#ifdef __cplusplus
} //namespace sharmonic
#endif

#endif
