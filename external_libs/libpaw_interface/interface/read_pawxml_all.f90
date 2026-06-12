subroutine read_pawxml_all(ecut,ecutpaw,gmet,ngfft,ngfftdg, &
        natom,ntypat,typat,ixc,xclevel,filename_list)
    use m_pawpsp
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod

    implicit none

    integer :: stat
    integer :: n3, i3, n2, i2
    integer :: it, ia
    integer :: iproj,imax

    real*8  :: ecut, ecutpaw !a coarse grid, and a fine grid for PAW
    real*8  :: gmet(3,3) !reciprocal space lattice vectors
    integer :: ngfft(3),ngfftdg(3)
    integer :: ixc, xclevel !functional; will be set externally in practice
    integer :: ntypat, natom
    integer :: typat(natom)
    integer :: nspden,nsppol!number of spin components for density and wavefunctions
    !for normal nspin = 1,2 calculations, nsppol is set to be same as nspden

    character(len=264) :: filename_list(ntypat)

    hyb_mixing = 0.0
    hyb_range_fock = 0.0

    call deallocate_arrays

    allocate(zion(ntypat),znucl(ntypat), nattyp(ntypat), lexexch(ntypat), lpawu(ntypat),stat=stat)
    if(stat/=0) then
        write(*,*) 'problem allocating ntypat sized arrays bundle 1. Stat ', stat
        call exit(1)
    endif
    allocate(pawrad(ntypat), pawtab(ntypat),epsatm(ntypat),stat=stat)
    if(stat/=0) then
        write(*,*) 'problem allocating ntypat sized arrays bundle 2. Stat ', stat
        call exit(1)
    endif
    ! not supporting dftu for now!
    lexexch = 0
    lpawu = 0

    ! Process energy cutoff
    call getcut(ecut,gmet,gsqcut,iboxcut,ngfft)
    call getcut(ecutpaw,gmet,gsqcutdg,iboxcut,ngfftdg)
    
    allocate(qgrid_ff(mqgrid),qgrid_vl(mqgrid),stat=stat)
    if(stat/=0) then
        write(*,*) 'problem allocating mqgrid'
        call exit(1)
    endif
    
    call generate_qgrid(gsqcut,qgrid_ff,mqgrid)
    call generate_qgrid(gsqcutdg,qgrid_vl,mqgrid)

    allocate(ffspl(mqgrid,2,lnmax), vlspl(mqgrid,2,ntypat))

    llmax = 0

    call paw_setup_free(pawsetup)
    call paw_setup_free(paw_setuploc)
    do it = 1, ntypat
        ! Read paw input files
        call rdpawpsxml(filename_list(it), pawsetup)
        call rdpawpsxml(filename_list(it), paw_setuploc)
        call pawpsp_read_header_xml(lloc, lmax, pspcod, pspxc,&
            & pawsetup, r2well, zion(it), znucl(it))
        call pawpsp_read_pawheader(pawpsp_header%basis_size,&
            &   lmax,pawpsp_header%lmn_size,&
            &   pawpsp_header%l_size, pawpsp_header%mesh_size,&
            &   pawpsp_header%pawver, pawsetup,&
            &   pawpsp_header%rpaw, pawpsp_header%rshp, pawpsp_header%shape_type)

        ! Process onsite information
        call pawtab_nullify(pawtab(it))
        call pawtab_set_flags(pawtab(it),has_tvale=1,has_vhnzc=1,has_vhtnzc=1,has_tproj=1)
        call pawpsp_17in(epsatm(it), ffspl, icoulomb, ipsp, hyb_mixing, ixc, lmax,&
                &       lnmax, pawpsp_header%mesh_size, mqgrid, mqgrid, pawpsp_header,&
                &       pawrad(it), pawtab(it), xcdev, qgrid_ff, qgrid_vl, usewvl, usexcnhat,&
                &       vlspl(:,:,it), xcccrc, xclevel, denpos, zion(it), znucl(it))
        call paw_setup_free(pawsetup)
        call paw_setup_free(paw_setuploc)

        llmax = max(lmax,llmax)
    enddo
contains
    subroutine generate_qgrid(gsqcut,qgrid,mqgrid)
        real*8  :: gsqcut
        real*8  :: qmax, dq
        real*8  :: qgrid(mqgrid)

        integer :: mqgrid, iq

        qmax = 1.2d0 * sqrt(gsqcut)
        dq = qmax/(1.0*(mqgrid-1))
        do iq = 1,mqgrid
            qgrid(iq) = (iq-1)*dq
        enddo
    end subroutine
end subroutine
