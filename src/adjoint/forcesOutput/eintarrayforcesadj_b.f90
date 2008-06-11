!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 2.2.4 (r2308) - 03/04/2008 10:03
!  
!  Differentiation of eintarrayforcesadj in reverse (adjoint) mode:
!   gradient, with respect to input variables: gammaconstant p
!                eint
!   of linear combination of output variables: p eint
!      ==================================================================
SUBROUTINE EINTARRAYFORCESADJ_B(rho, p, pb, k, eint, eintb, correctfork&
&  , kk)
  USE inputphysics
  USE constants
  USE flowvarrefstate
  IMPLICIT NONE
  INTEGER(KIND=INTTYPE), INTENT(IN) :: kk
!
!      ******************************************************************
!      *                                                                *
!      * EintArray computes the internal energy per unit mass from the  *
!      * given density and pressure (and possibly turbulent energy) for *
!      * the given kk elements of the arrays.                           *
!      * For a calorically and thermally perfect gas the well-known     *
!      * expression is used; for only a thermally perfect gas, cp is a  *
!      * function of temperature, curve fits are used and a more        *
!      * complex expression is obtained.                                *
!      *                                                                *
!      ******************************************************************
!
!
!      Subroutine arguments.
!
  REAL(KIND=REALTYPE), DIMENSION(kk) :: rho, p, k
  REAL(KIND=REALTYPE), DIMENSION(kk) :: pb
  REAL(KIND=REALTYPE), DIMENSION(kk) :: eint
  REAL(KIND=REALTYPE), DIMENSION(kk) :: eintb
!!$       real(kind=realType), dimension(*), intent(in)  :: rho, p, k
!!$       real(kind=realType), dimension(*), intent(out) :: eint
  LOGICAL, INTENT(IN) :: correctfork
!
!      Local parameter.
!
  REAL(KIND=REALTYPE), PARAMETER :: twothird=two*third
!
!      Local variables.
!
  INTEGER(KIND=INTTYPE) :: i, nn, mm, ii, start
  REAL(KIND=REALTYPE) :: ovgm1, factk, pp, t, t2, scale
!!$           ! Loop over the number of elements of the array
!!$
!!$           do i=1,kk
!!$
!!$             ! Compute the dimensional temperature.
!!$
!!$             pp = p(i)
!!$             if( correctForK ) pp = pp - twoThird*rho(i)*k(i)
!!$             t = Tref*pp/(RGas*rho(i))
!!$
!!$             ! Determine the case we are having here.
!!$
!!$             if(t <= cpTrange(0)) then
!!$
!!$               ! Temperature is less than the smallest temperature
!!$               ! in the curve fits. Use extrapolation using
!!$               ! constant cv.
!!$
!!$               eint(i) = scale*(cpEint(0) + cv0*(t - cpTrange(0)))
!!$
!!$             else if(t >= cpTrange(cpNparts)) then
!!$
!!$               ! Temperature is larger than the largest temperature
!!$               ! in the curve fits. Use extrapolation using
!!$               ! constant cv.
!!$
!!$               eint(i) = scale*(cpEint(cpNparts) &
!!$                       +        cvn*(t - cpTrange(cpNparts)))
!!$
!!$             else
!!$
!!$               ! Temperature is in the curve fit range.
!!$               ! First find the valid range.
!!$
!!$               ii    = cpNparts
!!$               start = 1
!!$               interval: do
!!$
!!$                 ! Next guess for the interval.
!!$
!!$                 nn = start + ii/2
!!$
!!$                 ! Determine the situation we are having here.
!!$
!!$                 if(t > cpTrange(nn)) then
!!$
!!$                   ! Temperature is larger than the upper boundary of
!!$                   ! the current interval. Update the lower boundary.
!!$
!!$                   start = nn + 1
!!$                   ii    = ii - 1
!!$
!!$                 else if(t >= cpTrange(nn-1)) then
!!$
!!$                   ! This is the correct range. Exit the do-loop.
!!$
!!$                   exit
!!$
!!$                 endif
!!$
!!$                 ! Modify ii for the next branch to search.
!!$
!!$                 ii = ii/2
!!$
!!$               enddo interval
!!$
!!$               ! Nn contains the correct curve fit interval.
!!$               ! Integrate cv to compute eint.
!!$
!!$               eint(i) = cpTempFit(nn)%eint0 - t
!!$               do ii=1,cpTempFit(nn)%nterm
!!$                 if(cpTempFit(nn)%exponents(ii) == -1_intType) then
!!$                   eint(i) = eint(i) &
!!$                           + cpTempFit(nn)%constants(ii)*log(t)
!!$                 else
!!$                   mm   = cpTempFit(nn)%exponents(ii) + 1
!!$                   t2   = t**mm
!!$                   eint(i) = eint(i) &
!!$                           + cpTempFit(nn)%constants(ii)*t2/mm
!!$                 endif
!!$               enddo
!!$
!!$               eint(i) = scale*eint(i)
!!$
!!$             endif
!!$
!!$             ! Add the turbulent energy if needed.
!!$
!!$             if( correctForK ) eint(i) = eint(i) + k(i)
!!$
!!$           enddo
!
!      ******************************************************************
!      *                                                                *
!      * Begin execution                                                *
!      *                                                                *
!      ******************************************************************
!
! Determine the cp model used in the computation.
  SELECT CASE  (cpmodel) 
  CASE (cpconstant) 
! Abbreviate 1/(gamma -1) a bit easier.
    ovgm1 = one/(gammaconstant-one)
    DO i=kk,1,-1
      pb(i) = pb(i) + ovgm1*eintb(i)/rho(i)
      eintb(i) = 0.0
    END DO
  END SELECT
!  gammaconstantb = 0.0
END SUBROUTINE EINTARRAYFORCESADJ_B
