!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 3.4 (r3375) - 10 Feb 2010 15:08
!
!  Differentiation of curveupre in forward (tangent) mode:
!   variations   of useful results: curveupre
!   with respect to varying inputs: *up3 *up2 *up1 *up0 *ret re
!
!      ******************************************************************
!      *                                                                *
!      * File:          curveFit.f90                                    *
!      * Author:        Georgi Kalitzin, Edwin van der Weide            *
!      * Starting date: 03-02-2004                                      *
!      * Last modified: 04-12-2005                                      *
!      *                                                                *
!      ******************************************************************
!
FUNCTION CURVEUPRE_D(re, red, curveupre)
  USE PARAMTURB
  IMPLICIT NONE
!
!      ******************************************************************
!      *                                                                *
!      * curveUpRe determines the value of the nonDimensional           *
!      * tangential velocity (made nonDimensional with the skin         *
!      * friction velocity) for the given Reynolds number.              *
!      * This data has been curve fitted with cubic splines.            *
!      *                                                                *
!      ******************************************************************
!
!
!      Function type.
!
  REAL(kind=realtype) :: curveupre
  REAL(kind=realtype) :: curveupre_d
!
!      Function arguments.
!
  REAL(kind=realtype), INTENT(IN) :: re
  REAL(kind=realtype), INTENT(IN) :: red
!
!      Local variables.
!
  INTEGER(kind=inttype) :: ii, nn, start
  REAL(kind=realtype) :: x, x2, x3, upre
  REAL(kind=realtype) :: xd, x2d, x3d, upred
  REAL(kind=realtype) :: arg1
  REAL(kind=realtype) :: arg1d
  INTRINSIC SQRT
!
!      ******************************************************************
!      *                                                                *
!      * Begin execution                                                *
!      *                                                                *
!      ******************************************************************
!
! Determine the situation we are dealing with.
  IF (re .LE. ret(0)) THEN
! Reynolds number is less than the smallest number in the curve
! fit. Use extrapolation.
    arg1d = red/ret(0)
    arg1 = re/ret(0)
    IF (arg1 .EQ. 0.0) THEN
      xd = 0.0
    ELSE
      xd = arg1d/(2.0*SQRT(arg1))
    END IF
    x = SQRT(arg1)
    upred = up0(1)*xd
    upre = x*up0(1)
  ELSE IF (re .GE. ret(nfit)) THEN
! Reynolds number is larger than the largest number in the curve
! fit. Set upRe to the largest value available.
    nn = nfit
    x = ret(nn) - ret(nn-1)
    x2 = x*x
    x3 = x*x2
    upre = up0(nn) + up1(nn)*x + up2(nn)*x2 + up3(nn)*x3
    upred = 0.0
  ELSE
! Reynolds number is in the range of the curve fits.
! First find the correct interval.
    ii = nfit
    start = 1
interval:DO 
! Next guess for the interval.
      nn = start + ii/2
! Determine the situation we are having here.
      IF (re .GT. ret(nn)) THEN
! Reynoldls number is larger than the upper boundary of
! the current interval. Update the lower boundary.
        start = nn + 1
        ii = ii - 1
      ELSE IF (re .GE. ret(nn-1)) THEN
        GOTO 100
      END IF
! This is the correct range. Exit the do-loop.
! Modify ii for the next branch to search.
      ii = ii/2
    END DO interval
! Compute upRe using the cubic polynomial for this interval.
 100 xd = red
    x = re - ret(nn-1)
    x2d = xd*x + x*xd
    x2 = x*x
    x3d = xd*x2 + x*x2d
    x3 = x*x2
    upred = up1(nn)*xd + up2(nn)*x2d + up3(nn)*x3d
    upre = up0(nn) + up1(nn)*x + up2(nn)*x2 + up3(nn)*x3
  END IF
! And set the function value.
  curveupre_d = upred
  curveupre = upre
END FUNCTION CURVEUPRE_D
