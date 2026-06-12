function pawxml_nproj(filename) result(nproj)
!This routine extracts the total number of projectors accounting for all channels, l, and m 
    use libpaw_mod, only : pawtab=>pawtab_read, filename_read

    implicit none

    character(len=264) :: filename
    integer :: nproj

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract nprojectors is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    nproj = pawtab%lmn_size
end function