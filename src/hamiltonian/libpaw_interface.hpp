/* -*- indent-tabs-mode: t -*- */

#ifndef HAMILTONIAN__LIBPAW_INTERFACE
#define HAMILTONIAN__LIBPAW_INTERFACE

// Copyright (C) 2019-2026 Lawrence Livermore National Security, LLC., Xavier Andrade, Alfredo A. Correa
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

extern "C"
{
	void fortran_main_(double&, double&, double*, double*, double*, double&, int*, int*,
	//                 ecut     ecutpaw  gmet     rprimd   gprimd   ucvol    ngfft ngfftdg
		int&, int&,  int*, double*, char*);
	//  natom ntypat typat xred     filename_list

	// TODO : last commented argument is the Epseudo. Need to extract later. 
	void read_pawxml_all_(double&, double&, double*, int*, int*,
	//                    ecut    ecutpaw gmet    ngfft ngfftdg
		int&, int&,  int*, int&,int&,   char*);
	//  natom ntypat typat ixc  xclevel filename_list

	// TODO : last commented argument is the Epseudo 
	void prepare_libpaw_(double&,double&,double*,double*,double*,double&,int*,int*,
	//                   ecut    ecutpaw gmet    rprimd  gprimd  ucvol   ngfft ngfftdg
		int&,int&,int*,double*,int&,int&,char*,int&,int&);  //,double*);
	//  natom ntypat typat xred ixc xclevel filename_list nspden nsppol

	void get_vloc_ncoret_(int*,   int&, int&, int&,  double*,double*,double*,
	//                    ngfftdg,nfft, natom,ntypat,rprimd, gprimd, gmet
		double&,double*,double*,   double*);
	//  ,ucvol, xred,   ncoretsum, vloc_sum

	void set_rhoij_(int&, int&,     int&,      int&,  int*,       double*);
	//              iatom nrhoijsel size_rhoij nspden rhoijselect rhoijp

	void calculate_dij_(int&, int&, int&, int&,   int&, int&,  double*, double&, double*, double*, double*, double&);
	//                  natom,ntypat,ixc, xclevel,nfft, nspden,xred,    ucvol,   gprimd,  vks,     vxc,     epawdc

	void get_dij_(int&, int&,    int&,  double*);
	//            iatom,size_dij,nspden,dij

	void get_nhat_(int&, int&, double*, int*, int&, int&, double*, double*, double&, double*, double*);
	//             natom,ntypat,xred,   ngfft,nfft,nspden,gprimd,  rprimd,  ucvol,   nhat,    nhatgr
}

#endif

#ifdef INQ_HAMILTONIAN_LIBPAW_INTERFACE_UNIT_TEST
#undef INQ_HAMILTONIAN_LIBPAW_INTERFACE_UNIT_TEST

#include <catch2/catch_all.hpp>

TEST_CASE(INQ_TEST_FILE, INQ_TEST_TAG) {

}
#endif


