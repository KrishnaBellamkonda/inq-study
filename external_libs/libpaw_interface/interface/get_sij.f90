subroutine get_sij(filename,lmn_size,sij)
    !pawtab only stores the upper triangle half of the sij matrix as it is symmetric
    use libpaw_mod, only : pawtab=>pawtab_read, filename_read
    implicit none
    integer :: lmn_size,i,j,k
    real*8  :: sij(lmn_size,lmn_size)
    character(len=264) :: filename

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract the overlap matrix sij is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    k=0
    do i=1, lmn_size
        do j=1, i
            k=k+1
            sij(i,j) = pawtab%sij(k)
            sij(j,i) = pawtab%sij(k)
        enddo
    enddo
!
!    do i=1, lmn_size
!        write(41,*) sij(i,:)
!    enddo
end subroutine