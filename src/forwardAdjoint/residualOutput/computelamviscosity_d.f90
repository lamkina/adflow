   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.6 (r4159) - 21 Sep 2011 10:11
   !
   !  Differentiation of computelamviscosity in forward (tangent) mode:
   !   variations   of useful results: *p
   !   with respect to varying inputs: *p *w
   !   Plus diff mem management of: p:in w:in rlv:in
   !
   !      ******************************************************************
   !      *                                                                *
   !      * File:          computeLamViscosity.f90                         *
   !      * Author:        Edwin van der Weide                             *
   !      * Starting date: 03-10-2003                                      *
   !      * Last modified: 06-12-2005                                      *
   !      *                                                                *
   !      ******************************************************************
   !
   SUBROUTINE COMPUTELAMVISCOSITY_D()
   USE FLOWVARREFSTATE
   USE BLOCKPOINTERS_D
   USE INPUTPHYSICS
   USE CONSTANTS
   USE ITERATION
   IMPLICIT NONE
   !
   !      ******************************************************************
   !      *                                                                *
   !      * computeLamViscosity computes the laminar viscosity ratio in    *
   !      * the owned cell centers of the given block. Sutherland's law is *
   !      * used. It is assumed that the pointes already point to the      *
   !      * correct block before entering this subroutine.                 *
   !      *                                                                *
   !      ******************************************************************
   !
   !
   !      Local parameter.
   !
   REAL(kind=realtype), PARAMETER :: twothird=two*third
   !
   !      Local variables.
   !
   INTEGER(kind=inttype) :: i, j, k
   REAL(kind=realtype) :: musuth, tsuth, ssuth, t
   LOGICAL :: correctfork
   !
   !      ******************************************************************
   !      *                                                                *
   !      * Begin execution                                                *
   !      *                                                                *
   !      ******************************************************************
   !
   ! Return immediately if no laminar viscosity needs to be computed.
   IF (.NOT.viscous) THEN
   RETURN
   ELSE
   ! Determine whether or not the pressure must be corrected
   ! for the presence of the turbulent kinetic energy.
   IF (kpresent) THEN
   IF (currentlevel .LE. groundlevel .OR. turbcoupled) THEN
   correctfork = .true.
   ELSE
   correctfork = .false.
   END IF
   ELSE
   correctfork = .false.
   END IF
   ! Compute the nonDimensional constants in sutherland's law.
   musuth = musuthdim/muref
   tsuth = tsuthdim/tref
   ssuth = ssuthdim/tref
   ! Substract 2/3 rho k, which is a part of the normal turbulent
   ! stresses, in case the pressure must be corrected.
   IF (correctfork) THEN
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   pd(i, j, k) = pd(i, j, k) - twothird*(wd(i, j, k, irho)*w(i&
   &              , j, k, itu1)+w(i, j, k, irho)*wd(i, j, k, itu1))
   p(i, j, k) = p(i, j, k) - twothird*w(i, j, k, irho)*w(i, j, &
   &              k, itu1)
   END DO
   END DO
   END DO
   END IF
   ! Loop over the owned cells of this block and compute the
   ! laminar viscosity ratio.
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   ! Compute the nonDimensional temperature and the
   ! nonDimensional laminar viscosity.
   t = p(i, j, k)/(rgas*w(i, j, k, irho))
   rlvd(i, j, k) = 0.0
   rlv(i, j, k) = musuth*((tsuth+ssuth)/(t+ssuth))*(t/tsuth)**&
   &            1.5_realType
   END DO
   END DO
   END DO
   ! Add the 2/3 rho k again to the pressure if the pressure was
   ! corrected earlier.
   IF (correctfork) THEN
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   pd(i, j, k) = pd(i, j, k) + twothird*(wd(i, j, k, irho)*w(i&
   &              , j, k, itu1)+w(i, j, k, irho)*wd(i, j, k, itu1))
   p(i, j, k) = p(i, j, k) + twothird*w(i, j, k, irho)*w(i, j, &
   &              k, itu1)
   END DO
   END DO
   END DO
   END IF
   END IF
   END SUBROUTINE COMPUTELAMVISCOSITY_D
