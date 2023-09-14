!        generated by tapenade     (inria, ecuador team)
!  tapenade 3.16 (develop) - 29 jan 2021 13:55
!
module actuatorregion_d
  use constants
  use communication, only : commtype, internalcommtype
  use actuatorregiondata
  implicit none
! ----------------------------------------------------------------------
!                                                                      |
!                    no tapenade routine below this line               |
!                                                                      |
! ----------------------------------------------------------------------

contains
!  differentiation of computeactuatorregionvolume in forward (tangent) mode (with options i4 dr8 r8):
!   variations   of useful results: actuatorregions.vollocal
!   with respect to varying inputs: *vol actuatorregions.vollocal
!   rw status of diff variables: *vol:in actuatorregions.vollocal:in-out
!   plus diff mem management of: vol:in
  subroutine computeactuatorregionvolume_d(nn, iregion)
    use blockpointers, only : ndom, vol, vold
    implicit none
! inputs
    integer(kind=inttype), intent(in) :: nn, iregion
! working
    integer(kind=inttype) :: iii
    integer(kind=inttype) :: i, j, k
! loop over the region for this block
    do iii=actuatorregions(iregion)%blkptr(nn-1)+1,actuatorregions(&
&       iregion)%blkptr(nn)
      i = actuatorregions(iregion)%cellids(1, iii)
      j = actuatorregions(iregion)%cellids(2, iii)
      k = actuatorregions(iregion)%cellids(3, iii)
! sum the volume of each cell within the region on this proc
      actuatorregionsd(iregion)%vollocal = actuatorregionsd(iregion)%&
&       vollocal + vold(i, j, k)
      actuatorregions(iregion)%vollocal = actuatorregions(iregion)%&
&       vollocal + vol(i, j, k)
    end do
  end subroutine computeactuatorregionvolume_d

  subroutine computeactuatorregionvolume(nn, iregion)
    use blockpointers, only : ndom, vol
    implicit none
! inputs
    integer(kind=inttype), intent(in) :: nn, iregion
! working
    integer(kind=inttype) :: iii
    integer(kind=inttype) :: i, j, k
! loop over the region for this block
    do iii=actuatorregions(iregion)%blkptr(nn-1)+1,actuatorregions(&
&       iregion)%blkptr(nn)
      i = actuatorregions(iregion)%cellids(1, iii)
      j = actuatorregions(iregion)%cellids(2, iii)
      k = actuatorregions(iregion)%cellids(3, iii)
! sum the volume of each cell within the region on this proc
      actuatorregions(iregion)%vollocal = actuatorregions(iregion)%&
&       vollocal + vol(i, j, k)
    end do
  end subroutine computeactuatorregionvolume

end module actuatorregion_d

