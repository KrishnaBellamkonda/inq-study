/* -*- indent-tabs-mode: t -*- */

// Copyright (C) 2019-2023 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <inq/inq.hpp>

int main(int argc, char ** argv){

	using namespace inq;
	using namespace inq::magnitude;
	using inq::vector3;
	
	inq::utils::match match(3.0e-4);

	{
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "water.xyz", systems::cell::orthorhombic(12.0_b, 11.0_b, 10.0_b).finite());
		
		auto & env = input::environment::global();
		auto parstates = env.comm().size();
		if(env.comm().size() == 3 or env.comm().size() == 5) parstates = 1;
		
		systems::electrons electrons(env.par().states(parstates), ions, options::electrons{}.cutoff(30.0_Ha));
		
		ground_state::initial_guess(ions, electrons);
		
		auto scf_options = options::ground_state{}.energy_tolerance(1.0e-9_Ha).broyden_mixing();
		auto result = ground_state::calculate(ions, electrons, options::theory{}.lda(), scf_options);
		
		match.check("total energy",        result.energy.total(),       -17.604152928274);
		match.check("kinetic energy",      result.energy.kinetic(),      12.055671278976);
		match.check("eigenvalues",         result.energy.eigenvalues(),  -4.066524514529);
		match.check("Hartree energy",      result.energy.hartree(),      21.255096093096);
		match.check("external energy",     result.energy.external(),    -50.405054484947);
		match.check("non-local energy",    result.energy.non_local(),     -2.732683697594);
		match.check("XC energy",           result.energy.xc(),           -4.762305356613);
		match.check("XC density integral", result.energy.nvxc(),         -5.494649797157);
		match.check("HF exchange energy",  result.energy.exact_exchange(),   0.000000000000);
		match.check("ion-ion energy",      result.energy.ion(),           6.985123238808);
		
		std::cout << result.dipole << std::endl;
		
		match.check("dipole x", result.dipole[0], -0.000304523);
		match.check("dipole y", result.dipole[1], -0.724304);
		match.check("dipole z", result.dipole[2], -2.78695e-05);
		
		electrons.save("h2o_restart");
	}


	{
		
		auto ions = systems::ions::parse(config::path::unit_tests_data() + "water.xyz", systems::cell::orthorhombic(12.0_b, 11.0_b, 10.0_b).finite());
		
		auto & env = input::environment::global();
		auto parstates = env.comm().size();
		if(env.comm().size() == 3 or env.comm().size() == 5) parstates = 1;
		
		systems::electrons electrons(env.par().states(parstates), ions, options::electrons{}.cutoff(30.0_Ha));

		// Propagation without perturbation
		{
			electrons.load("h2o_restart");
			
			std::vector<double> energy;
			std::vector<double> dipole;

			auto output = [&](auto data){
				energy.push_back(data.energy().total());
				dipole.push_back(data.dipole()[1]);
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(30).dt(0.055_atomictime));

			assert(energy.size() == 31ul);
			
			match.check("ETRS: energy step   0", energy[0],   -17.604152928110);
			match.check("ETRS: energy step  10", energy[10],  -17.604152928110);
			match.check("ETRS: energy step  20", energy[20],  -17.604152928110);
			match.check("ETRS: energy step  30", energy[30],  -17.604152928110);

			match.check("ETRS: dipole step   0", dipole[0],   -0.724267424533);
			match.check("ETRS: dipole step  10", dipole[10],  -0.724267425292);
			match.check("ETRS: dipole step  20", dipole[20],  -0.724267430201);
			match.check("ETRS: dipole step  30", dipole[30],  -0.724267436727);
		}
		
		// Propagation without perturbation
		{
			electrons.load("h2o_restart");
			
			std::vector<double> energy;
			std::vector<double> dipole;
				
			auto output = [&](auto data){
				energy.push_back(data.energy().total());
				dipole.push_back(data.dipole()[1]);
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(10).dt(0.1_atomictime).crank_nicolson());
			
			match.check("CN: energy step  0", energy[0], -17.604152917249);
			match.check("CN: energy step  1", energy[1], -17.604152917238);
			match.check("CN: energy step  2", energy[2], -17.604152917219);
			match.check("CN: energy step  3", energy[3], -17.604152917209);
			match.check("CN: energy step  4", energy[4], -17.604152917067);
			match.check("CN: energy step  5", energy[5], -17.604152917063);
			match.check("CN: energy step  6", energy[6], -17.604152916323);
			match.check("CN: energy step  7", energy[7], -17.604152916317);
			match.check("CN: energy step  8", energy[8], -17.604152916330);
			match.check("CN: energy step  9", energy[9], -17.604152916313);
			
			match.check("CN: dipole step  0", dipole[0], -0.724267424533);
			match.check("CN: dipole step  1", dipole[1], -0.724268354897);
			match.check("CN: dipole step  2", dipole[2], -0.724268979425);
			match.check("CN: dipole step  3", dipole[3], -0.724269361383);
			match.check("CN: dipole step  4", dipole[4], -0.724269728414);
			match.check("CN: dipole step  5", dipole[5], -0.724269528087);
			match.check("CN: dipole step  6", dipole[6], -0.724276940299);
			match.check("CN: dipole step  7", dipole[7], -0.724277232453);
			match.check("CN: dipole step  8", dipole[8], -0.724277380605);
			match.check("CN: dipole step  9", dipole[9], -0.724276950753);
		}
		
		{
			electrons.load("h2o_restart");
			
			auto kick = perturbations::kick{ions.cell(), {0.1, 0.0, 0.0}};
			
			long nsteps = 70;
			
			gpu::array<double, 1> time(nsteps + 1);
			gpu::array<double, 1> dip(nsteps + 1);
			gpu::array<double, 1> en(nsteps + 1);		
			
			auto output = [&](auto data){
				
				auto iter = data.iter();
				
				time[iter] = data.time();
				dip[iter] = data.dipole()[0];
				en[iter] = data.energy().total();			
				
				if(data.root() and data.every(50)){
					auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), dip({0, iter - 1}));  
					
					std::ofstream file("spectrum.dat");
					
					for(int ifreq = 0; ifreq < spectrum.size(); ifreq++){
						file << ifreq*in_atomic_units(0.01_eV) << '\t' << real(spectrum[ifreq]) << '\t' << imag(spectrum[ifreq]) << std::endl;
					}
				}
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(nsteps).dt(0.055_atomictime), kick);
			
			match.check("ETRS length kick: dipole step   0", dip[0],  -0.000309741032);
			match.check("ETRS length kick: dipole step  10", dip[10], -0.353225338482);
			match.check("ETRS length kick: dipole step  20", dip[20], -0.517982705117);
			match.check("ETRS length kick: dipole step  30", dip[30], -0.554109329493);
			match.check("ETRS length kick: dipole step  40", dip[40], -0.506630768030);
			match.check("ETRS length kick: dipole step  50", dip[50], -0.411072088843);
			match.check("ETRS length kick: dipole step  60", dip[60], -0.293501486218);
			match.check("ETRS length kick: dipole step  70", dip[70], -0.174726584012);
			
			match.check("ETRS length kick: energy step   0", en[0],   -17.563616880918);
			match.check("ETRS length kick: energy step  10", en[10],  -17.563606697534);
			match.check("ETRS length kick: energy step  20", en[20],  -17.563614886423);
			match.check("ETRS length kick: energy step  30", en[30],  -17.563621242161);
			match.check("ETRS length kick: energy step  40", en[40],  -17.563628740525);
			match.check("ETRS length kick: energy step  50", en[50],  -17.563634809017);
			match.check("ETRS length kick: energy step  60", en[60],  -17.563641013280);
			match.check("ETRS length kick: energy step  70", en[70],  -17.563647956213);
		}

		{
			electrons.load("h2o_restart");
			
			auto kick1 = perturbations::kick{ions.cell(), {0.06, 0.0, 0.0}, perturbations::gauge::velocity};
			auto kick2 = perturbations::kick{ions.cell(), {0.04, 0.0, 0.0}, perturbations::gauge::velocity};
			
			long nsteps = 30;
			
			gpu::array<double, 1> time(nsteps + 1);
			gpu::array<double, 1> dip(nsteps + 1);
			gpu::array<double, 1> en(nsteps + 1);	
			
			auto output = [&](auto data){

				auto iter = data.iter();
			
				time[iter] = data.time();
				dip[iter] = data.dipole()[0];
				en[iter] = data.energy().total();			
				
				if(data.root() and data.every(50)){
					auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), dip({0, iter - 1}));  
					
					std::ofstream file("spectrum.dat");
					
					for(int ifreq = 0; ifreq < spectrum.size(); ifreq++){
						file << ifreq*in_atomic_units(0.01_eV) << '\t' << real(spectrum[ifreq]) << '\t' << imag(spectrum[ifreq]) << std::endl;
					}
				}
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(nsteps).dt(0.055_atomictime), kick1 + kick2);

			match.check("ETRS velocity kick: dipole step   0", dip[0],  -0.000309741032);
			match.check("ETRS velocity kick: dipole step  10", dip[10], -0.352667764228);
			match.check("ETRS velocity kick: dipole step  20", dip[20], -0.517468633127);
			match.check("ETRS velocity kick: dipole step  30", dip[30], -0.553770289583);

			match.check("ETRS velocity kick: energy step   0", en[0],   -17.563920909871);
			match.check("ETRS velocity kick: energy step  10", en[10],  -17.563906701155);
			match.check("ETRS velocity kick: energy step  20", en[20],  -17.563911157818);
			match.check("ETRS velocity kick: energy step  30", en[30],  -17.563913885907);
		}

		{
			electrons.load("h2o_restart");

			auto perts = perturbations::blend{};
			perts.add(perturbations::kick{ions.cell(), {0.06, 0.0, 0.0}});
			perts.add(perturbations::kick{ions.cell(), {0.04, 0.0, 0.0}});
			
			long nsteps = 30;
			
			gpu::array<double, 1> time(nsteps + 1);
			gpu::array<double, 1> dip(nsteps + 1);
			gpu::array<double, 1> en(nsteps + 1);		
			
			auto output = [&](auto data){

				auto iter = data.iter();
			
				time[iter] = data.time();
				dip[iter] = data.dipole()[0];
				en[iter] = data.energy().total();			
				
				if(data.root() and data.every(50)){
					auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), dip({0, iter - 1}));  
					
					std::ofstream file("spectrum.dat");
					
					for(int ifreq = 0; ifreq < spectrum.size(); ifreq++){
						file << ifreq*in_atomic_units(0.01_eV) << '\t' << real(spectrum[ifreq]) << '\t' << imag(spectrum[ifreq]) << std::endl;
					}
				}
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.lda(), options::real_time{}.num_steps(nsteps).dt(0.055_atomictime), perts);

			match.check("ETRS length kick: dipole step   0", dip[0],  -0.000309741032);
			match.check("ETRS length kick: dipole step  10", dip[10], -0.353225338482);
			match.check("ETRS length kick: dipole step  20", dip[20], -0.517982705117);
			match.check("ETRS length kick: dipole step  30", dip[30], -0.554109329493);
			
			match.check("ETRS length kick: energy step   0", en[0],   -17.563616880918);
			match.check("ETRS length kick: energy step  10", en[10],  -17.563606697534);
			match.check("ETRS length kick: energy step  20", en[20],  -17.563614886423);
			match.check("ETRS length kick: energy step  30", en[30],  -17.563621242161);
		}

		// Propagation changing the TDDFT functional without perturbation
		{
			electrons.load("h2o_restart");
			
			std::vector<double> energy;
			std::vector<double> dipole;
				
			auto output = [&](auto data){
				energy.push_back(data.energy().total());
				dipole.push_back(data.dipole()[1]);
			};
			
			auto opts = options::real_time{}.num_steps(10).dt(0.1_atomictime).crank_nicolson();
			opts = opts.ground_state_theory(options::theory{}.lda());
			real_time::propagate<>(ions, electrons, output, options::theory{}.pbe(), opts);

			//these are the LDA values
			match.check("CN: energy step  0", energy[0], -17.604152917249);
			match.check("CN: energy step  1", energy[1], -17.604152917238);
			match.check("CN: energy step  2", energy[2], -17.604152917219);
			match.check("CN: energy step  3", energy[3], -17.604152917209);
			match.check("CN: energy step  4", energy[4], -17.604152917067);
			match.check("CN: energy step  5", energy[5], -17.604152917063);
			match.check("CN: energy step  6", energy[6], -17.604152916323);
			match.check("CN: energy step  7", energy[7], -17.604152916317);
			match.check("CN: energy step  8", energy[8], -17.604152916330);
			match.check("CN: energy step  9", energy[9], -17.604152916313);
			
			match.check("CN: dipole step  0", dipole[0], -0.724267424533);
			match.check("CN: dipole step  1", dipole[1], -0.724268354897);
			match.check("CN: dipole step  2", dipole[2], -0.724268979425);
			match.check("CN: dipole step  3", dipole[3], -0.724269361383);
			match.check("CN: dipole step  4", dipole[4], -0.724269728414);
			match.check("CN: dipole step  5", dipole[5], -0.724269528087);
			match.check("CN: dipole step  6", dipole[6], -0.724276940299);
			match.check("CN: dipole step  7", dipole[7], -0.724277232453);
			match.check("CN: dipole step  8", dipole[8], -0.724277380605);
			match.check("CN: dipole step  9", dipole[9], -0.724276950753);
			
		}

		// Propagation changing the TDDFT functional with perturbation
		{
			electrons.load("h2o_restart");
			
			auto kick = perturbations::kick{ions.cell(), {0.1, 0.0, 0.0}};
			
			long nsteps = 70;
			
			gpu::array<double, 1> time(nsteps + 1);
			gpu::array<double, 1> dip(nsteps + 1);
			gpu::array<double, 1> en(nsteps + 1);
			
			auto output = [&](auto data){
				
				auto iter = data.iter();
				
				time[iter] = data.time();
				dip[iter] = data.dipole()[0];
				en[iter] = data.energy().total();			
				
				if(data.root() and data.every(50)){
					auto spectrum = observables::spectrum(20.0_eV, 0.01_eV, time({0, iter - 1}), dip({0, iter - 1}));  
					
					std::ofstream file("spectrum.dat");
					
					for(int ifreq = 0; ifreq < spectrum.size(); ifreq++){
						file << ifreq*in_atomic_units(0.01_eV) << '\t' << real(spectrum[ifreq]) << '\t' << imag(spectrum[ifreq]) << std::endl;
					}
				}
			};
			
			real_time::propagate<>(ions, electrons, output, options::theory{}.pbe(),
														 options::real_time{}.num_steps(nsteps).dt(0.055_atomictime).ground_state_theory(options::theory{}.lda()), kick);

			//these are not the LDA values
			match.check("ETRS length kick: dipole step   0", dip[0],  -0.000309796245);
			match.check("ETRS length kick: dipole step  10", dip[10], -0.353525175330);
			match.check("ETRS length kick: dipole step  20", dip[20], -0.519304090087);
			match.check("ETRS length kick: dipole step  30", dip[30], -0.556387104315);
			match.check("ETRS length kick: dipole step  40", dip[40], -0.509274513784);
			match.check("ETRS length kick: dipole step  50", dip[50], -0.413238589926);
			match.check("ETRS length kick: dipole step  60", dip[60], -0.294147858693);
			match.check("ETRS length kick: dipole step  70", dip[70], -0.173079826166);
			
			match.check("ETRS length kick: energy step   0", en[0],   -17.563616880759);
			match.check("ETRS length kick: energy step  10", en[10],  -17.563503799073);
			match.check("ETRS length kick: energy step  20", en[20],  -17.563399658106);
			match.check("ETRS length kick: energy step  30", en[30],  -17.563438115613);
			match.check("ETRS length kick: energy step  40", en[40],  -17.563501509341);
			match.check("ETRS length kick: energy step  50", en[50],  -17.563551564499);
			match.check("ETRS length kick: energy step  60", en[60],  -17.563564642959);
			match.check("ETRS length kick: energy step  70", en[70],  -17.563520334792);
		}

		
	}
	
	fftw_cleanup(); //required for valgrind
		
	return match.fail();

}

