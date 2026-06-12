/* -*- indent-tabs-mode: t -*- */

#ifndef SHARMONIC_REFERENCE
#define SHARMONIC_REFERENCE

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

#include <sharmonic.h>

#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
#include <boost/math/special_functions/spherical_harmonic.hpp>
#endif

#ifdef HAVE_GSL
#include <gsl/gsl_sf_legendre.h>
#endif

#ifdef __cplusplus
namespace sharmonic {
#endif

///////////////////////////////////////////////////////////////////

#if defined(__cplusplus) && defined(HAVE_SPH_LEGENDRE)
namespace ref_std {

inline auto angular_real(int const ll, int const mm, double theta, double phi) {
	
	// This is the general implementation using the C++ standard library (that doesn't work on the GPU and it slower)
	
	auto sh = std::sph_legendre(ll, abs(mm), theta);
	if(mm > 0){
		return internal::powsign(mm)*std::sqrt(2.0)*cos(mm*phi)*sh;
	} else if (mm == 0) {
		return sh;
	} else {
		return internal::powsign(mm)*std::sqrt(2.0)*sin(-mm*phi)*sh;
	}
}

//////////////////////////////////////////////////////////////////

inline auto normalized_cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_std::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

inline auto cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_std::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex angular_complex(int const ll, int const mm, double theta, double phi) {
	using namespace std::complex_literals;

	// This is the general implementation using the C++ standard library (that doesn't work on the GPU and it slower)
	auto leg = std::sph_legendre(ll, abs(mm), theta);
	if(mm < 0) leg *= internal::powsign(mm); //only positive mm values are supported
	return std::polar(leg, mm*phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex normalized_cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_std::angular_complex<Complex>(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_std::angular_complex<Complex>(ll, mm, theta, phi);
}

} //namespace ref_std
#endif



///////////////////////////////////////////////////////////////////

#if defined(__cplusplus) && defined(HAVE_BOOST_SPHERICAL_HARMONIC)
namespace ref_boost {

inline auto angular_real(int const ll, int const mm, double theta, double phi) {
	
	// this is a reference implementation using boost.

	if(phi < 0.0) phi += 2.0*M_PI; //boost wants phi in the [0, 2pi) range

	auto sh = boost::math::spherical_harmonic(ll, abs(mm), theta, phi);
	if(mm < 0) return sqrt(2.0)*pow(-1.0, mm)*imag(sh);
	if(mm > 0) return sqrt(2.0)*pow(-1.0, mm)*real(sh);
	return real(sh);
}

///////////////////////////////////////////////////////////////////

inline auto normalized_cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_boost::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

inline auto cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_boost::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex angular_complex(int const ll, int const mm, double theta, double phi) {
	
	// this is a reference implementation using boost.
	if(phi < 0.0) phi += 2.0*M_PI; //boost wants phi in the [0, 2pi) range
	return boost::math::spherical_harmonic(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex normalized_cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_boost::angular_complex<Complex>(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_boost::angular_complex<Complex>(ll, mm, theta, phi);
}

} //namespace ref_boost
#endif

//////////////////////////////////////////////////////////////////

#if defined(__cplusplus) && defined(HAVE_GSL)
namespace ref_gsl {

inline auto angular_real(int const ll, int const mm, double theta, double phi) {
	
	auto leg = gsl_sf_legendre_sphPlm(ll, abs(mm), cos(theta));
	if(mm > 0){
		return internal::powsign(mm)*std::sqrt(2.0)*cos(mm*phi)*leg;
	} else if (mm == 0) {
		return leg;
	} else {
		return internal::powsign(mm)*std::sqrt(2.0)*sin(-mm*phi)*leg;
	}
}

//////////////////////////////////////////////////////////////////

inline auto normalized_cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_gsl::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

inline auto cartesian_real(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_gsl::angular_real(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex angular_complex(int const ll, int const mm, double theta, double phi) {
	using namespace std::complex_literals;

	auto leg = gsl_sf_legendre_sphPlm(ll, abs(mm), cos(theta));
	if(mm < 0) leg *= internal::powsign(mm); //only positive mm values are supported
	return std::polar(leg, mm*phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex normalized_cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double theta, phi;
	internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);
	return ref_gsl::angular_complex<Complex>(ll, mm, theta, phi);
}

///////////////////////////////////////////////////////////////////

template <typename Complex = std::complex<double>>
inline Complex cartesian_complex(int const ll, int const mm, double xx, double yy, double zz) {

	if(ll == 0) return 1.0/sqrt(4.0*M_PI);
	double rr, theta, phi;
	internal::cartesian_to_angular(xx, yy, zz, &rr, &theta, &phi);
	if(rr < 1e-15) return 0.0;
	return ref_gsl::angular_complex<Complex>(ll, mm, theta, phi);
}

} //namespace ref_gsl
#endif

#ifdef __cplusplus
} //namespace sharmonic
#endif

#endif
