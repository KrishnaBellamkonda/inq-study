/* -*- indent-tabs-mode: t -*- */

#ifndef HAMILTONIAN__PAW
#define HAMILTONIAN__PAW

// Copyright (C) 2019-2026 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include <options/electrons.hpp>
#include <states/orbital_set.hpp>
#include <basis/field_set.hpp>
#include <basis/real_space.hpp>
#include <basis/field.hpp>
#include <utils/vec_field.hpp>
#include <hamiltonian/libpaw_interface.hpp>

// Debugging for prepare paw remove later
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <iomanip>
// Debugging for prepare paw remove above later

namespace inq {
namespace hamiltonian {

class paw {

private:

	std::vector<int> ngfft_;
	std::vector<int> ngfftdg_;
	std::vector<int> typat_;
	std::vector<int> size_dij_;
	std::vector<double> gmet_;
	std::vector<double> rprimd_;
	std::vector<double> gprimd_;
	std::vector<double> xred_;
	std::vector<double> dij_;
	double ucvol_;
	double ecut_;
	double ecutpaw_;
	double epawdc_;
	int natom_;
	int ntypat_;
	int nfft_;
	int ixc_; 
	int xclevel_;
	int nspden_; 
	int nsppol_;
	char* filename_list_;
	basis::real_space density_basis_;
	basis::field<basis::real_space, double> vloc_sum_rs_;
	basis::field<basis::real_space, double> ncoret_sum_rs_;
	basis::field_set<basis::real_space, double> nhat_fs_rs_;
	bool libpaw_prepped_ = false;
	// basis::field_set<basis::real_space, double> nhatgr_rs_;

public: 

	template<class T> void what(T&&) = delete;

	template <class ions_type, class atomic_pot_type>
	paw(const ions_type & ions, const basis::real_space & density_basis, const int & nspin, const atomic_pot_type & atomic_pot, const options::electrons & conf): 
		nspden_(nspin),
		nsppol_(1),
		density_basis_(density_basis),
		vloc_sum_rs_(density_basis),
		ncoret_sum_rs_(density_basis),
		nhat_fs_rs_(density_basis, nspden_)//,
		// nhatgr_rs_(density_basis) // This term needs to have 3 times the number of grid points since it's a gradient
	{

		ecut_    = std::max(conf.cutoff_value() / 2.0, 10.0); // cutoff_value / 2.0 is an arbitrary choice for now.
		ecutpaw_ = ecut_;
		gmet_.resize(9);
		rprimd_.resize(9);
		gprimd_.resize(9);
		ngfft_.resize(3);
		ngfftdg_.resize(3);

		if(!ions.cell().is_orthogonal())
			std::runtime_error("Error: Current implementation of PAW only works with orthorhombic cells for now");
		if(nspden_ != 1)
			std::runtime_error("Error: Current implementation of PAW only works with spin = 1 for now");
		for(auto ivec = 0; ivec < 3; ivec++) {
			for(auto icoord = 0; icoord < 3; icoord++) {
				rprimd_[ivec * 3 + icoord] = ions.cell().lattice(ivec)[icoord];
				// Abinit reciprocal lattices vectors aren't multipled by 2*pi for historical reasons.
				gprimd_[ivec * 3 + icoord] = ions.cell().reciprocal(ivec)[icoord] / (2 * M_PI);
			}
		}

		// https://docs.abinit.org/topics/documents/geometry.pdf
		// gmet_ is the metric tensor in reciprocal space
		mattmat(gprimd_, gmet_);

		natom_  = ions.size();
		ntypat_ = ions.species_list().size();
		ucvol_  = ions.cell().volume();

		ngfft_[0] = density_basis.sizes()[0];
		ngfft_[1] = density_basis.sizes()[1];
		ngfft_[2] = density_basis.sizes()[2];
		ngfftdg_  = ngfft_;

		typat_.resize(natom_, -1);
		size_dij_.resize(natom_);
		filename_list_ = new char[ntypat_ * 264];
		for(int i = 0; i < ntypat_ * 264; i++) {
			filename_list_[i] = ' ';
		}

		int current_type = 1;
		std::map<std::string, int> species_to_type;
		for(int iatom = 0; iatom < natom_; iatom++) {
			auto symbol = ions.species(iatom).symbol();

			if(species_to_type.find(symbol) == species_to_type.end()) {
				species_to_type[symbol] = current_type;
				auto filename           = ions.species(iatom).file_path();
				for(long unsigned int j = 0; j < filename.size(); j++) {
					filename_list_[264 * (current_type - 1) + j] = filename[j];
				}
				++current_type;
			}
			typat_[iatom] = species_to_type[symbol];

			auto num_lm = atomic_pot.pseudo_for_element(ions.species(iatom)).num_projectors_lm();
			size_dij_[iatom] = num_lm*(num_lm+1)/2;
		}

		//focusing on orthorhombic cell for now.
		// TODO: need to figure out row-major and col-major transformations in calculating how matrices are flatten and reduced/fractional coordinates for generalized cell.
		// https://docs.abinit.org/variables/basic/#xred
		xred_ = calculate_xred(ions);

		nfft_ = ngfft_[0]*ngfft_[1]*ngfft_[2];
	}

	void prepare_libpaw(const options::theory & inter) {
		// https://docs.abinit.org/variables/basic/#ixc
		// can only use internal Abinit XC functionals for now
		if(inter.exchange() == XC_LDA_X && inter.correlation() == XC_LDA_C_PZ) {
			ixc_ = 2;  // 1981 Perdew-Zunger LDA
			// ixc_ = 7; // 1992 Perdew-Wang LDA
			xclevel_ = 1;
		} else if(inter.exchange() == XC_GGA_X_PBE && inter.correlation() == XC_GGA_C_PBE) {
			// 1996 Perdew-Burke-Ernzerhof GGA
			ixc_     = 11;
			xclevel_ = 2;
		} else {
			throw std::runtime_error("Only PZ LDA and PBE GGA is implemented in PAW for now");
		}

		read_pawxml_all_(ecut_, ecutpaw_, gmet_.data(), ngfft_.data(), ngfftdg_.data(), natom_, ntypat_, typat_.data(), ixc_, xclevel_, filename_list_);

		prepare_libpaw_(ecut_, ecutpaw_, gmet_.data(), rprimd_.data(), gprimd_.data(), ucvol_, ngfft_.data(), ngfftdg_.data(), natom_, ntypat_, typat_.data(), xred_.data(), ixc_, xclevel_, filename_list_, nspden_, nsppol_);

		libpaw_prepped_ = true;
	}


	std::vector<double> convertColumnMajorToRowMajor3D(const std::vector<double>& colMajorArray, int dim1, int dim2, int dim3) {
		std::vector<double> rowMajorArray(dim1 * dim2 * dim3);

		for(auto i = 0; i < dim1; ++i) {
			for(auto j = 0; j < dim2; ++j) {
				for(auto k = 0; k < dim3; ++k) {
					// Column-major index: k * dim1 * dim2 + j * dim1 + i
					// Row-major index: i * dim2 * dim3 + j * dim3 + k
					rowMajorArray[(i * dim2 + j) * dim3 + k] = colMajorArray[(k * dim2 + j) * dim1 + i];
				}
			}
		}

		return rowMajorArray;
	}

	std::vector<double> convertRowMajorToColumnMajor3D(const std::vector<double>& rowMajorArray, int dim1, int dim2, int dim3) {
		std::vector<double> colMajorArray(dim1 * dim2 * dim3);

		for(auto i = 0; i < dim1; ++i) {
			for(auto j = 0; j < dim2; ++j) {
				for(auto k = 0; k < dim3; ++k) {
					// Column-major index: k * dim1 * dim2 + j * dim1 + i
					// Row-major index: i * dim2 * dim3 + j * dim3 + k
					colMajorArray[(k * dim2 + j) * dim1 + i] = rowMajorArray[(i * dim2 + j) * dim3 + k];
				}
			}
		}

		return colMajorArray;
	}

	void mattmat(std::vector<double>& mat_in, std::vector<double>& mat_out) {
		// calculates G = A^T A for 3 by 3 matrix
		// G_ij = sum_k A_ki A_kj
		mat_out[0] = mat_in[0] * mat_in[0] + mat_in[1] * mat_in[1] + mat_in[2] * mat_in[2];
		mat_out[1] = mat_in[0] * mat_in[3] + mat_in[1] * mat_in[4] + mat_in[2] * mat_in[5];
		mat_out[2] = mat_in[0] * mat_in[6] + mat_in[1] * mat_in[7] + mat_in[2] * mat_in[8];
		mat_out[3] = mat_in[3] * mat_in[0] + mat_in[4] * mat_in[1] + mat_in[5] * mat_in[2];
		mat_out[4] = mat_in[3] * mat_in[3] + mat_in[4] * mat_in[4] + mat_in[5] * mat_in[5];
		mat_out[5] = mat_in[3] * mat_in[6] + mat_in[4] * mat_in[7] + mat_in[5] * mat_in[8];
		mat_out[6] = mat_in[6] * mat_in[0] + mat_in[7] * mat_in[1] + mat_in[8] * mat_in[2];
		mat_out[7] = mat_in[6] * mat_in[3] + mat_in[7] * mat_in[4] + mat_in[8] * mat_in[5];
		mat_out[8] = mat_in[6] * mat_in[6] + mat_in[7] * mat_in[7] + mat_in[8] * mat_in[8];
	}

	template <typename KpointType, class projectors_type, typename OccupationType>
	void calculate_rhoij(states::orbital_set<basis::real_space, complex> const & phi, KpointType const & kpoint, const projectors_type & projectors_all, OccupationType const & occupations) {

		int                 nrhoijsel;
		std::vector<int>    rhoijselect;
		std::vector<double> rhoijp;  // nspden=1 for now

		auto nprojs  = projectors_all.nprojs();
		assert(nprojs > 0 && nprojs == natom_);
		auto projections_all = projectors_all.calculate_projections(phi, kpoint);

		auto rhoij = projectors_all.density_matrix(phi, kpoint, occupations);

		for(auto iatom = 0; iatom < nprojs; iatom++) {

			auto nlm                 = projectors_all.nlm()[iatom];
			auto size_upper_triangle = nlm * (nlm + 1) / 2;

			rhoijselect.resize(size_upper_triangle, 0);
			rhoijp.resize(size_upper_triangle, 0);

			int ind = -1;
			for(auto iproj = 0; iproj < nlm; iproj++) {
				for(auto jproj = 0; jproj < iproj + 1; jproj++) {
					ind         = ind + 1;
					rhoijp[ind] = rhoij[iatom][jproj][iproj];
				}
			}

			for(auto i = 0; i < size_upper_triangle; i++) {
				rhoijselect[i] = 0.0;  // RHS Fortran indexing
			}

			nrhoijsel = 0;
			for(auto i = 0; i < size_upper_triangle; i++) {
				if(fabs(rhoijp[i]) > 1e-10) {
					rhoijselect[nrhoijsel] = i + 1;  // RHS Fortran indexing
					rhoijp[nrhoijsel]      = rhoijp[i];
					nrhoijsel += 1;
				}
			}

			int ia_fortran = iatom + 1;
			int nspden     = nspden_;
			set_rhoij_(ia_fortran, nrhoijsel, size_upper_triangle, nspden, rhoijselect.data(), rhoijp.data()); // This Fortran function has to be updated to handle multiple k-points
		}
	}

	template <class ions_type>
	auto calculate_xred(const ions_type & ions) {
		std::vector<double> xred;
		xred.resize(natom_ * 3);
		for (int iatom = 0; iatom < natom_; iatom++){
			xred[iatom * 3 + 0] = ions.positions()[iatom][0] / rprimd_[0];
			xred[iatom * 3 + 1] = ions.positions()[iatom][1] / rprimd_[4];
			xred[iatom * 3 + 2] = ions.positions()[iatom][2] / rprimd_[8];
		}
		return xred;
	}

	template <class ions_type>
	void calculate_vloc_ncoret(const ions_type & ions) { // Static atoms
		auto xred = calculate_xred(ions);
		std::vector<double> vloc_sum(nfft_ * nspden_);
		std::vector<double> ncoret_sum(nfft_ * nspden_);
		get_vloc_ncoret_(ngfftdg_.data(), nfft_, natom_, ntypat_, rprimd_.data(), gprimd_.data(), gmet_.data(), ucvol_, xred.data(), ncoret_sum.data(), vloc_sum.data());
		vloc_sum       = convertColumnMajorToRowMajor3D(vloc_sum, ngfft_[0], ngfft_[1], ngfft_[2]);
		ncoret_sum     = convertColumnMajorToRowMajor3D(ncoret_sum, ngfft_[0], ngfft_[1], ngfft_[2]);
		vloc_sum_rs_   = utils::vec_to_field(vloc_sum,   density_basis_);
		ncoret_sum_rs_ = utils::vec_to_field(ncoret_sum, density_basis_);
	}

	void calculate_nhat() {
		std::vector<double> nhat_(nfft_ * nspden_);
		std::vector<double> nhatgr_(3 * nfft_ * nspden_);
		get_nhat_(natom_, ntypat_, xred_.data(), ngfft_.data(), nfft_, nspden_, gprimd_.data(), rprimd_.data(), ucvol_, nhat_.data(), nhatgr_.data());
		nhat_       = convertColumnMajorToRowMajor3D(nhat_,ngfft_[0],ngfft_[1],ngfft_[2]);
		nhatgr_     = convertColumnMajorToRowMajor3D(nhatgr_,ngfft_[0],ngfft_[1],ngfft_[2]);
		nhat_fs_rs_ = utils::vec_to_field_set(nhat_, density_basis_, nspden_); 
		// nhatgr_rs_  = utils::vec_to_field(nhatgr_, density_basis_, nspden_); 
	}

	void calculate_dij(std::vector<double> const & vtrial, std::vector<double> const & vxc) {
		// Depends on rhoij^a, vks, vxc, and choice of XC functional (ix,xclevel)
		assert((long)vtrial.size() == nfft_ * nspden_ && vtrial.size() == vxc.size());

		// Convert from row to column major to pass into libpaw
		auto vtrial_fortran = convertRowMajorToColumnMajor3D(vtrial, ngfft_[0], ngfft_[1], ngfft_[2]);
		auto vxc_fortran    = convertRowMajorToColumnMajor3D(vxc, ngfft_[0], ngfft_[1], ngfft_[2]);

		calculate_dij_(natom_, ntypat_, ixc_, xclevel_, nfft_, nspden_, xred_.data(), ucvol_,
							gprimd_.data(), vtrial_fortran.data(), vxc_fortran.data(), epawdc_);
	}

	void get_dij(int iatom, std::vector<double> & dij) {
		int i_fortran = iatom + 1;
		assert((long)dij.size() == size_dij_[iatom]);
		get_dij_(i_fortran, size_dij_[iatom], nspden_, dij.data());
	}

	auto & vloc_sum() const {
		return vloc_sum_rs_;
	}

	auto & vloc_sum() {
		return vloc_sum_rs_;
	}

	auto & ncoret_sum() const {
		return ncoret_sum_rs_;
	}

	auto & ncoret_sum() {
		return ncoret_sum_rs_;
	}

	auto & nhat() const {
		return nhat_fs_rs_;
	}

	auto & nhat() {
		return nhat_fs_rs_;
	}

	void check_libpaw_prepped() {
		if(!libpaw_prepped_) throw std::runtime_error("libpaw not prepared. Please provide functional information vis paw.prepare_libpaw(functional)");
	}
	
};
}
}
#endif

#ifdef INQ_HAMILTONIAN_PAW_UNIT_TEST
#undef INQ_HAMILTONIAN_PAW_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {
	// Debugging and the values to check against for the following tests were done against a modified version of the OPAW-(TD)DFT code:
	// https://github.com/minhchemucla/OPAW-TDDFT.git
	using namespace inq;
	using namespace inq::magnitude;
	using namespace Catch::literals;
	using Catch::Approx;

	parallel::communicator comm{boost::mpi3::environment::get_world_instance()};

	SECTION("Calculate rhoij, nhat, vloc, and ncoret isolated C"){
		if(comm.size() == 1){
			auto cell = systems::cell::cubic(10.0_b);
			systems::ions ions(cell);
			ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {5.0_b, 5.0_b, 5.0_b});

			systems::electrons electrons(ions,  input::kpoints::grid({1,1,1}, false),options::electrons{}.cutoff(100.0_Ha).extra_states(0).temperature(0.0_K));
			auto functional = options::theory{}.lda();
			electrons.paw().prepare_libpaw(functional);
			ground_state::initial_guess(ions, electrons);

			hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.states(), electrons.atomic_pot(), ions);

			int iphi = 0;
			for(auto & phi : electrons.kpin()) {
				phi.fill(1.0);
				electrons.calculate_comp_charge(phi, phi.kpoint(), ham.projectors_all(), electrons.occupations()[iphi]);
				auto rhoij = ham.projectors_all().density_matrix(phi, phi.kpoint(), electrons.occupations()[iphi]);

				auto max_nlm = ham.projectors_all().max_nlm();
				auto nprojs = ham.projectors_all().nprojs();
				CHECK(max_nlm == 8);
				CHECK(nprojs  == 1);
				CHECK(rhoij[0][0][0] == Approx(143.207513192342020));
				CHECK(rhoij[0][1][0] == Approx(4.6862148745742482 ));
				CHECK(rhoij[0][0][1] == Approx(4.6862148745742482 ));
				CHECK(rhoij[0][1][1] == Approx(0.1533481684105892 ));
				for(int iproj = 2; iproj < max_nlm; iproj++){
					for(int jproj = 2; jproj < 8; jproj++){
						CHECK(rhoij[0][iproj][jproj] == Approx(0.0).margin(1e-6));
					}
				}

				auto total_nhat = electrons.comp_charge();
				CHECK(operations::integral_abs(total_nhat) == Approx(2.8725508042262233));

				electrons.paw().calculate_vloc_ncoret(ions);
				auto vloc_total = electrons.paw().vloc_sum();
				CHECK(operations::integral_abs(vloc_total) == Approx(232.6903285065255886));

				auto ncoret_total = electrons.paw().ncoret_sum();
				CHECK(operations::integral_abs(ncoret_total) == Approx(0.6832673638844491));
			}

			auto density = observables::density::calculate(electrons);
			auto nhat = electrons.spin_comp_charge().value();
			observables::density::normalize(density, nhat, electrons.states().num_electrons());
			CHECK(operations::integral_sum_add(density, nhat) == Approx(4.0));
		}
	}

	SECTION("Calculate rhoij, nhat, vloc, and ncoret methane"){
		if(comm.size()==1){
			auto cell = systems::cell::cubic(10.0_b);
			systems::ions ions(cell);
			
			ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {2.20210452600000_b, 5.7109405980_b, 5.0_b});
			ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {2.87608365200000_b, 3.80456610500000_b, 5.0_b});
			ions.insert(ionic::species("H"), {2.87611846100000_b, 6.66411496700000_b, 6.65096194500000_b});
			ions.insert(ionic::species("H"), {2.87611846100000_b, 6.66411496700000_b, 3.34903805500000_b});
			ions.insert(ionic::species("H"), {0.18009771800000_b, 5.71096550500000_b, 5.0_b});

			systems::electrons electrons(ions,  input::kpoints::grid({1,1,1}, false), options::electrons{}.cutoff(100.0_Ha).extra_states(0).temperature(0.0_K));
			auto functional = options::theory{}.lda();
			electrons.paw().prepare_libpaw(functional);
			ground_state::initial_guess(ions, electrons);

			hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.states(), electrons.atomic_pot(), ions);

			int iphi = 0;
			for(auto & phi : electrons.kpin()) {
				phi.fill(1.0);
				electrons.calculate_comp_charge(phi, phi.kpoint(), ham.projectors_all(), electrons.occupations()[iphi]);
				auto rhoij = ham.projectors_all().density_matrix(phi, phi.kpoint(), electrons.occupations()[iphi]);

				auto max_nlm = ham.projectors_all().max_nlm();
				auto nprojs = ham.projectors_all().nprojs();
				CHECK(max_nlm == 8);
				CHECK(nprojs  == 5);

				/*
				The l=0 projectors are in principle the only non-orthogonal projectors to a wavefunction that
				is a constant since a constant wavefunction is a factor of Y_0^0, however, because a discrete grid
				is used, the l>0 projectors might have values that are "significantly" non-zero in magnitude unless 
				the atom is exactly on a grid point like the last example. In the limit of infinite grid points, 
				these l>0 terms should go to zero. For this test, we'll only check the l=0 terms.
				*/
				CHECK(rhoij[0][0][0] == Approx(286.390596388625965));
				CHECK(rhoij[0][1][0] == Approx(9.3709773592960524 ));
				CHECK(rhoij[0][0][1] == Approx(9.3709773592960524 ));
				CHECK(rhoij[0][1][1] == Approx(0.3066274443916302 ));

				CHECK(rhoij[1][0][0] == Approx(7781.4642090443057896 ));
				CHECK(rhoij[1][1][0] == Approx(-4489.2570002611710152));
				CHECK(rhoij[1][0][1] == Approx(-4489.2570002611710152));
				CHECK(rhoij[1][1][1] == Approx(2589.9275345853070576 ));

				CHECK(rhoij[2][0][0] == Approx(7773.7880680758717062 ));
				CHECK(rhoij[2][1][0] == Approx(-4484.3801755902195509));
				CHECK(rhoij[2][0][1] == Approx(-4484.3801755902195509));
				CHECK(rhoij[2][1][1] == Approx(2586.8553893062339739 ));

				CHECK(rhoij[3][0][0] == Approx(7773.7880680758662493 ));
				CHECK(rhoij[3][1][0] == Approx(-4484.3801755902177320));
				CHECK(rhoij[3][0][1] == Approx(-4484.3801755902177320));
				CHECK(rhoij[3][1][1] == Approx(2586.8553893062339739 ));

				CHECK(rhoij[4][0][0] == Approx(7778.4938013462297022 ));
				CHECK(rhoij[4][1][0] == Approx(-4487.4266187752582482));
				CHECK(rhoij[4][0][1] == Approx(-4487.4266187752582482));
				CHECK(rhoij[4][1][1] == Approx(2588.8042303778170208 ));

				auto total_nhat = electrons.comp_charge();
				CHECK(operations::integral_abs(total_nhat) == Approx(7.2582754802027836));

				electrons.paw().calculate_vloc_ncoret(ions);
				auto vloc_total = electrons.paw().vloc_sum();
				CHECK(operations::integral_abs(vloc_total) == Approx(423.5062121260317554));

				auto ncoret_total = electrons.paw().ncoret_sum();
				CHECK(operations::integral_abs(ncoret_total) == Approx(0.6839431274775114));
			}

			auto density = observables::density::calculate(electrons);
			auto nhat = electrons.spin_comp_charge().value();
			observables::density::normalize(density, nhat, electrons.states().num_electrons());
			CHECK(operations::integral_sum_add(density, nhat) == Approx(8.0));
		}
	}

	SECTION("Methane with PAW hamiltonian from initial guess"){
		if(comm.size()==1){
			auto cell = systems::cell::cubic(10.0_b);
			systems::ions ions(cell);
			ions.insert(ionic::species("C").pseudo_file(config::path::unit_tests_data() + "C_PAW.xml"), {-2.79789547400000_b,  7.109405980E-001_b, 0.0_b});
			ions.insert(ionic::species("H").pseudo_file(config::path::unit_tests_data() + "H_PAW.xml"), {-2.12391634800000_b, -1.19543389500000_b, 0.0_b});
			ions.insert(ionic::species("H"), {-2.12388153900000_b, 1.66411496700000_b, 1.65096194500000_b});
			ions.insert(ionic::species("H"), {-2.12388153900000_b, 1.66411496700000_b, -1.65096194500000_b});
			ions.insert(ionic::species("H"), {-4.81990228200000_b, 0.71096550500000_b, 0.000000000000000_b});

			systems::electrons electrons(ions,  input::kpoints::grid({1,1,1}, false), options::electrons{}.cutoff(45.0_Ha).extra_states(3).temperature(0.0_K));
			auto functional = options::theory{}.lda();
			electrons.paw().prepare_libpaw(functional);
			ground_state::initial_guess(ions, electrons);

			hamiltonian::ks_hamiltonian<double> ham(electrons.states_basis(), electrons.states(), electrons.atomic_pot(), ions);

			hamiltonian::self_consistency<perturbations::none> sc(functional, electrons);

			inq::ground_state::results res;
			sc.update_ionic_fields(electrons.states_comm(), ions, electrons.atomic_pot(), electrons.paw_option());
			auto ked = std::optional<decltype(observables::kinetic_energy_density(electrons))>{};
			sc.update_hamiltonian(ham, res.energy, electrons.spin_density(), ked, electrons.spin_comp_charge());
			sc.update_hamiltonian_nl(ham, electrons);

			int ngrid = electrons.states_basis().sizes()[0] * electrons.states_basis().sizes()[1] * electrons.states_basis().sizes()[2];
			for(auto & phi : electrons.kpin()) {
				auto sphi = ham.overlap_operator()(phi);
				complex phi_s_phi = 0;
				for(int ip = 0; ip < ngrid; ip++){
					phi_s_phi += electrons.spin_density().basis().volume_element() * (conj(phi.matrix()[ip][0]) * sphi.matrix()[ip][0]);
				}
				CHECK(Approx(phi_s_phi.real()).epsilon(1e-8) == 1.0);

				// <1|H|1>
				auto hphi = ham(phi);
				complex phi_h_phi = 0;
				for(int ip = 0; ip < ngrid; ip++){
					phi_h_phi += electrons.spin_density().basis().volume_element() * (conj(phi.matrix()[ip][0]) * hphi.matrix()[ip][0]);
				}
				CHECK(Approx(phi_h_phi.real()).epsilon(1e-8) == 44.1674788784626244);
			}
		}
	}
}
#endif


