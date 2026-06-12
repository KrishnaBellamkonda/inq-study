/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifdef HAVE_SHARMONIC_CONFIG_H
#include <sharmonic_config.h>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <sharmonic.h>
#include <sharmonic_reference.h>

#include <complex>
#include <chrono>
#include <cassert>

#include <iostream>
#include <fstream>
#include <vector>

#ifdef ENABLE_GPU
#include <thrust/complex.h>
using complex = thrust::complex<double>;
#else
#include <complex>
using complex = std::complex<double>;
#endif

#define SMALL_NUMBER 5.0e-15

#ifdef ENABLE_GPU

#define KERNEL_REPS 16

#ifdef ENABLE_CUDA
#define cudahip(x) cuda ## x
#else
#define cudahip(x) hip ## x
#endif

#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }
template <typename Code>
inline void gpu_assert(Code code, const char *file, int line) {
	if (code == cudahip(Success)) return;
	fprintf(stderr,"GPU error: %s %s %d\n", cudahip(GetErrorString)(code), file, line);
	exit(code);
}

__global__ void kernel_cart_real(double * array, int ll, int mm, double xx, double yy, double zz) {
	auto val = 0.0;
	for(int iter = 0; iter < KERNEL_REPS; iter++) val += sharmonic::cartesian_real(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}
__global__ void kernel_cart_comp(complex * array, int ll, int mm, double xx, double yy, double zz) {
	auto val = complex{0.0, 0.0};
	for(int iter = 0; iter < KERNEL_REPS; iter++) val += sharmonic::cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}

__global__ void kernel_norm_real(double * array, int ll, int mm, double xx, double yy, double zz) {
	auto val = 0.0;
	for(int iter = 0; iter < 10; iter++) val += sharmonic::normalized_cartesian_real(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}
__global__ void kernel_norm_comp(complex * array, int ll, int mm, double xx, double yy, double zz) {
	auto val = complex{0.0, 0.0};
	for(int iter = 0; iter < KERNEL_REPS; iter++) val += sharmonic::normalized_cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}

__global__ void kernel_ang_real(double * array, int ll, int mm, double theta, double phi) {
	auto val = 0.0;
	for(int iter = 0; iter < KERNEL_REPS; iter++) val += sharmonic::angular_real(ll, mm, theta, phi + SMALL_NUMBER*iter);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}
__global__ void kernel_ang_comp(complex * array, int ll, int mm, double theta, double phi) {
	auto val = complex{0.0, 0.0};
	for(int iter = 0; iter < KERNEL_REPS; iter++) val += sharmonic::angular_complex<complex>(ll, mm, theta, phi + SMALL_NUMBER*iter);
	array[blockIdx.x*blockDim.x + threadIdx.x] = val;
}
#endif

void gpu_sync() {
#ifdef ENABLE_GPU
	gpu_error_check(cudahip(DeviceSynchronize)());
#endif
}

auto benchmark() {

	auto reps    = 1000;
	auto npoints = 100;
#ifdef ENABLE_GPU
	auto gpu_size = 1024;
	auto block_size = 1024;
#endif

	auto points = (double *) malloc(sizeof(double)*npoints*3);

	srand48(0xFC769E60887BB49Cl);
	for(int ip = 0; ip < npoints; ip++) {
		points[3*ip + 0] = 2.0*(drand48() - 0.5);
		points[3*ip + 1] = 2.0*(drand48() - 0.5);
		points[3*ip + 2] = 2.0*(drand48() - 0.5);		
	}

	auto dval = 0.0;
	auto zval = complex{0.0, 0.0};
	
	auto av_sh_re_cart = 0.0;
	auto av_sh_co_cart = 0.0;
	auto av_std_cart   = 0.0;
	auto av_boost_cart = 0.0;
	auto av_gsl_cart   = 0.0;
	
	auto av_sh_re_norm = 0.0;
	auto av_sh_co_norm = 0.0;
	auto av_std_norm   = 0.0;
	auto av_boost_norm = 0.0;
	auto av_gsl_norm   = 0.0;
	
	auto av_sh_re_ang  = 0.0;
	auto av_sh_co_ang  = 0.0;
	auto av_std_ang    = 0.0;
	auto av_boost_ang  = 0.0;
	auto av_gsl_ang    = 0.0;

#ifdef ENABLE_GPU
	double * darray;
	complex *zarray;
	gpu_error_check(cudahip(Malloc)((void**) &darray, reps*gpu_size*sizeof(double)));
	gpu_error_check(cudahip(Malloc)((void**) &zarray, reps*gpu_size*sizeof(complex)));
	gpu_sync();
#endif

	for(int ll = 1; ll <= SHARMONIC_MAX_L; ll++) {
 
		auto sh_re_cart = 0.0;
		auto sh_co_cart = 0.0;
		auto std_cart   = 0.0;
		auto boost_cart = 0.0;
		auto gsl_cart   = 0.0;
		
		auto sh_re_norm = 0.0;
		auto sh_co_norm = 0.0;
		auto std_norm   = 0.0;
		auto boost_norm = 0.0;
		auto gsl_norm   = 0.0;
		
		auto sh_re_ang  = 0.0;
		auto sh_co_ang  = 0.0;
		auto std_ang    = 0.0;
		auto boost_ang  = 0.0;
		auto gsl_ang    = 0.0;
		
		for(int mm = -ll; mm <= ll; mm++) {
			
			for(int ip = 0; ip < npoints; ip++) {

				auto xx = points[3*ip + 0];
				auto yy = points[3*ip + 1];
				auto zz = points[3*ip + 2];

				{
					dval = 0.0;
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_cart_real<<<reps*gpu_size/block_size, block_size>>>(darray, ll, mm, xx, yy, zz);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) dval += sharmonic::cartesian_real(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_re_cart += elapsed_seconds.count();
				}

				{
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_cart_comp<<<reps*gpu_size/block_size, block_size>>>(zarray, ll, mm, xx, yy, zz);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_co_cart += elapsed_seconds.count();
				}

#ifdef HAVE_SPH_LEGENDRE
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_std::cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					std_cart += elapsed_seconds.count();
				}
#else
				std_cart += HUGE_VAL;
#endif

#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_boost::cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					boost_cart += elapsed_seconds.count();
				}
#else
				boost_cart += HUGE_VAL;
#endif

#ifdef HAVE_GSL
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_gsl::cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					gsl_cart += elapsed_seconds.count();
				}
#else
				gsl_cart += HUGE_VAL;
#endif
				
				sharmonic::internal::normalize(&xx, &yy, &zz);
				
				{
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_norm_real<<<reps*gpu_size/block_size, block_size>>>(darray, ll, mm, xx, yy, zz);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) dval += sharmonic::normalized_cartesian_real(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_re_norm += elapsed_seconds.count();
				}

				{
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_norm_comp<<<reps*gpu_size/block_size, block_size>>>(zarray, ll, mm, xx, yy, zz);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::normalized_cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_co_norm += elapsed_seconds.count();
				}

#ifdef HAVE_SPH_LEGENDRE
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_std::normalized_cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					std_norm += elapsed_seconds.count();
				}
#else
				std_cart += HUGE_VAL;
#endif
				
#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_boost::normalized_cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					boost_norm += elapsed_seconds.count();
				}
#else
				boost_norm += HUGE_VAL;
#endif

#ifdef HAVE_GSL
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_gsl::normalized_cartesian_complex<complex>(ll, mm, xx + SMALL_NUMBER*iter, yy - SMALL_NUMBER*iter, zz);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					gsl_norm += elapsed_seconds.count();
				}
#else
				gsl_norm += HUGE_VAL;
#endif
				double theta, phi;
				sharmonic::internal::normalized_cartesian_to_angular(xx, yy, zz, &theta, &phi);

				{
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_ang_real<<<reps*gpu_size/block_size, block_size>>>(darray, ll, mm, theta, phi);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) dval += sharmonic::angular_real(ll, mm, theta, phi + SMALL_NUMBER*iter);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_re_ang  += elapsed_seconds.count();
				}

				{
					gpu_sync();
					auto iter_start_time = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_GPU
					kernel_ang_comp<<<reps*gpu_size/block_size, block_size>>>(zarray, ll, mm, theta, phi);
					gpu_error_check(cudahip(PeekAtLastError)());
					gpu_sync();
#else
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::angular_complex<complex>(ll, mm, theta, phi + SMALL_NUMBER*iter);
#endif
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					sh_co_ang  += elapsed_seconds.count();
				}

#ifdef HAVE_SPH_LEGENDRE
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_std::angular_complex<complex>(ll, mm, theta, phi + SMALL_NUMBER*iter);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					std_ang  += elapsed_seconds.count();
				}
#else
				std_cart += HUGE_VAL;
#endif

#ifdef HAVE_BOOST_SPHERICAL_HARMONIC
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_boost::angular_complex<complex>(ll, mm, theta, phi + SMALL_NUMBER*iter);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					boost_ang  += elapsed_seconds.count();
				}
#else
				boost_ang += HUGE_VAL;
#endif

#ifdef HAVE_GSL
				{
					auto iter_start_time = std::chrono::high_resolution_clock::now();
					for(int iter = 0; iter < reps; iter++) zval += sharmonic::ref_gsl::angular_complex<complex>(ll, mm, theta, phi + SMALL_NUMBER*iter);
					auto new_time = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> elapsed_seconds = new_time - iter_start_time;
					gsl_ang  += elapsed_seconds.count();
				}
#else
				gsl_ang += HUGE_VAL;
#endif
				
			}

		}

	
		double num = reps*npoints*(2.0*ll + 1.0)*1e-6;
		double gpu_num = num;
#ifdef ENABLE_GPU
		gpu_num *= gpu_size*KERNEL_REPS;
#endif
		
		av_sh_re_cart  += sh_re_cart/gpu_num;
		av_sh_co_cart  += sh_co_cart/gpu_num;
		av_std_cart    +=	std_cart/num;
		av_boost_cart  +=	boost_cart/num;
		av_gsl_cart    +=	gsl_cart/num;
		
		av_sh_re_norm  +=	sh_re_norm/gpu_num;
		av_sh_co_norm  +=	sh_co_norm/gpu_num;
		av_std_norm    +=	std_norm/num;
		av_boost_norm  +=	boost_norm/num;
		av_gsl_norm    +=	gsl_norm/num;
		
		av_sh_re_ang   +=	sh_re_ang/gpu_num;
		av_sh_co_ang   +=	sh_co_ang/gpu_num;
		av_std_ang     +=	std_ang/num;
		av_boost_ang   +=	boost_ang/num;
		av_gsl_ang     +=	gsl_ang/num;
		
		std::cout << ll 
							<< "\t\t" << gpu_num/sh_re_cart << '\t' << gpu_num/sh_co_cart << '\t' << num/std_cart << '\t' << num/boost_cart << '\t' << num/gsl_cart
							<< "\t\t" << gpu_num/sh_re_norm << '\t' << gpu_num/sh_co_norm << '\t' << num/std_norm << '\t' << num/boost_norm << '\t' << num/gsl_norm
							<< "\t\t" << gpu_num/sh_re_ang  << '\t' << gpu_num/sh_co_ang  << '\t' << num/std_ang  << '\t' << num/boost_ang  << '\t' << num/gsl_ang
							<< std::endl;
	}

	//dummy print to prevent optimization
	if(zval.real() == dval) std::cout << dval << '\t' << zval << std::endl;
	
	double numl = SHARMONIC_MAX_L;
	
	std::cout << "# average\n"
						<< "# cart " << numl/av_sh_re_cart << '\t' << numl/av_sh_co_cart << '\t' << numl/av_std_cart << '\t' << numl/av_boost_cart << '\t' << numl/av_gsl_cart << "\n"
						<< "# norm " << numl/av_sh_re_norm << '\t' << numl/av_sh_co_norm << '\t' << numl/av_std_norm << '\t' << numl/av_boost_norm << '\t' << numl/av_gsl_norm << "\n"
						<< "# ang  " << numl/av_sh_re_ang  << '\t' << numl/av_sh_co_ang  << '\t' << numl/av_std_ang  << '\t' << numl/av_boost_ang  << '\t' << numl/av_gsl_ang  << "\n"
						<< std::endl;
	
	free(points);
}

int main() {
	benchmark();
}
