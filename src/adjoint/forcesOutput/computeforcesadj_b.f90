!        Generated by TAPENADE     (INRIA, Tropics team)
!  Tapenade 2.2.4 (r2308) - 03/04/2008 10:03
!  
!  Differentiation of computeforcesadj in reverse (adjoint) mode:
!   gradient, with respect to input variables: cdadj cladj machadj
!                alphaadj xadj wadj betaadj cmzadj cmyadj machcoefadj
!                cmxadj
!   of linear combination of output variables: cdadj cladj cmzadj
!                cmyadj cmxadj
!
!     ******************************************************************
!     *                                                                *
!     * File:          computeForcesAdj.f90                            *
!     * Author:        C.A.(Sandy) Mader, Andre C. Marta               *
!     *                Seongim Choi
!     * Starting date: 12-14-2007                                      *
!     * Last modified: 12-27-2007                                      *
!     *                                                                *
!     ******************************************************************
!
SUBROUTINE COMPUTEFORCESADJ_B(xadj, xadjb, wadj, wadjb, padj, iibeg, &
&  iiend, jjbeg, jjend, i2beg, i2end, j2beg, j2end, mm, cfxadj, cfyadj, &
&  cfzadj, cmxadj, cmxadjb, cmyadj, cmyadjb, cmzadj, cmzadjb, yplusmax, &
&  refpoint, cladj, cladjb, cdadj, cdadjb, nn, level, sps, cfpadj, &
&  cmpadj, righthanded, secondhalo, alphaadj, alphaadjb, betaadj, &
&  betaadjb, machadj, machadjb, machcoefadj, machcoefadjb, prefadj, &
&  rhorefadj, pinfdimadj, rhoinfdimadj, rhoinfadj, pinfadj, murefadj, &
&  timerefadj, pinfcorradj)
  USE blockpointers
  USE inputtimespectral
  USE bctypes
  USE inputphysics
  USE communication
  USE flowvarrefstate
  IMPLICIT NONE
!!$      
!(xAdj, &
!         iiBeg,iiEnd,jjBeg,jjEnd,i2Beg,i2End,j2Beg,j2End, &
!         mm,cFxAdj,cFyAdj,cFzAdj, &
!         cMxAdj,cMyAdj,cMzAdj,yplusMax,refPoint,CLAdj,CDAdj,  &
!        nn,level,sps,cFpAdj,cMpAdj)
!
!     ******************************************************************
!     *                                                                *
!     * Computes the Force coefficients for the current configuration  *
!     * for the finest grid level and specified time instance using the*
!     * auxiliar routines modified for tapenade. This code calculates  *
!     * the result for a single boundary subface and requires an       *
!     * outside driver to loop over mm subfaces and nn domains to      *
!     * calculate the total forces and moments.                        *
!     *                                                                *
!     ******************************************************************
!
! ie,je,ke
! procHalo(currentLevel)%nProcSend, myID
! equations
! nTimeIntervalsSpectral!nTimeInstancesMax
! EulerWall, ...
!nw
!
!     Subroutine arguments.
!
  INTEGER(KIND=INTTYPE), INTENT(IN) :: mm, nn, level, sps
  INTEGER(KIND=INTTYPE), INTENT(IN) :: iibeg, iiend, jjbeg, jjend
  INTEGER(KIND=INTTYPE), INTENT(IN) :: i2beg, i2end, j2beg, j2end
  REAL(KIND=REALTYPE), DIMENSION(3) :: refpoint
  REAL(KIND=REALTYPE) :: yplusmax
  REAL(KIND=REALTYPE), DIMENSION(3) :: cfpadj, cmpadj, cfvadj, cmvadj
  REAL(KIND=REALTYPE), DIMENSION(0:ie, 0:je, 0:ke, 3) :: xadj
  REAL(KIND=REALTYPE), DIMENSION(0:ie, 0:je, 0:ke, 3) :: xadjb
  REAL(KIND=REALTYPE), DIMENSION(0:ib, 0:jb, 0:kb, nw) :: wadj
  REAL(KIND=REALTYPE), DIMENSION(0:ib, 0:jb, 0:kb, nw) :: wadjb
  REAL(KIND=REALTYPE), DIMENSION(0:ib, 0:jb, 0:kb) :: padj
  REAL(KIND=REALTYPE), DIMENSION(0:ib, 0:jb, 0:kb) :: padjb
! notice the range of x dim is set 1:2 which corresponds to 1/il
  REAL(KIND=REALTYPE), DIMENSION(ntimeintervalsspectral) :: cladj, cdadj&
&  , cfxadj, cfyadj, cfzadj, cmxadj, cmyadj, cmzadj
  REAL(KIND=REALTYPE), DIMENSION(ntimeintervalsspectral) :: cladjb, &
&  cdadjb, cmxadjb, cmyadjb, cmzadjb
  REAL(KIND=REALTYPE), DIMENSION(3) :: veldirfreestreamadj
  REAL(KIND=REALTYPE), DIMENSION(3) :: veldirfreestreamadjb
  REAL(KIND=REALTYPE), DIMENSION(3) :: liftdirectionadj
  REAL(KIND=REALTYPE), DIMENSION(3) :: liftdirectionadjb
  REAL(KIND=REALTYPE), DIMENSION(3) :: dragdirectionadj
  REAL(KIND=REALTYPE), DIMENSION(3) :: dragdirectionadjb
  REAL(KIND=REALTYPE) :: machadj, machcoefadj, uinfadj, pinfcorradj
  REAL(KIND=REALTYPE) :: machadjb, machcoefadjb, uinfadjb, pinfcorradjb
  REAL(KIND=REALTYPE), DIMENSION(nw) :: winfadj
  REAL(KIND=REALTYPE), DIMENSION(nw) :: winfadjb
  REAL(KIND=REALTYPE) :: prefadj, rhorefadj
  REAL(KIND=REALTYPE) :: pinfdimadj, rhoinfdimadj
  REAL(KIND=REALTYPE) :: rhoinfadj, pinfadj
  REAL(KIND=REALTYPE) :: murefadj, timerefadj
  REAL(KIND=REALTYPE) :: alphaadj, betaadj
  REAL(KIND=REALTYPE) :: alphaadjb, betaadjb
!
!     Local variables.
!
  REAL(KIND=REALTYPE), DIMENSION(2, iibeg:iiend, jjbeg:jjend, 3) :: &
&  siadj
  REAL(KIND=REALTYPE), DIMENSION(2, iibeg:iiend, jjbeg:jjend, 3) :: &
&  siadjb
! notice the range of y dim is set 1:2 which corresponds to 1/jl
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, 2, jjbeg:jjend, 3) :: &
&  sjadj
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, 2, jjbeg:jjend, 3) :: &
&  sjadjb
! notice the range of z dim is set 1:2 which corresponds to 1/kl
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, jjbeg:jjend, 2, 3) :: &
&  skadj
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, jjbeg:jjend, 2, 3) :: &
&  skadjb
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, jjbeg:jjend, 3) :: normadj
  REAL(KIND=REALTYPE), DIMENSION(iibeg:iiend, jjbeg:jjend, 3) :: &
&  normadjb
!add to allow for scaling!
  REAL(KIND=REALTYPE), DIMENSION(3) :: cfpadjout, cfvadjout
  REAL(KIND=REALTYPE), DIMENSION(3) :: cfpadjoutb, cfvadjoutb
  REAL(KIND=REALTYPE), DIMENSION(3) :: cmpadjout, cmvadjout
  REAL(KIND=REALTYPE), DIMENSION(3) :: cmpadjoutb, cmvadjoutb
  LOGICAL, INTENT(IN) :: righthanded, secondhalo
  INTEGER(KIND=INTTYPE) :: i, j, k, l, kk
!
!     ******************************************************************
!     *                                                                *
!     * Begin execution.                                               *
!     *                                                                *
!     ******************************************************************
!
!===============================================================
! Compute the forces.
!      call the initialization routines to calculate the effect of Mach and alpha
  CALL ADJUSTINFLOWANGLEFORCESADJ(alphaadj, betaadj, veldirfreestreamadj&
&                            , liftdirectionadj, dragdirectionadj)
  CALL PUSHREAL8(machcoefadj)
  CALL PUSHREAL8ARRAY(liftdirectionadj, 3)
  CALL PUSHREAL8ARRAY(veldirfreestreamadj, 3)
  CALL CHECKINPUTPARAMFORCESADJ(veldirfreestreamadj, liftdirectionadj, &
&                          dragdirectionadj, machadj, machcoefadj)
  CALL PUSHREAL8(rhorefadj)
  CALL PUSHREAL8(prefadj)
  CALL REFERENCESTATEFORCESADJ(machadj, machcoefadj, uinfadj, prefadj, &
&                         rhorefadj, pinfdimadj, rhoinfdimadj, rhoinfadj&
&                         , pinfadj, murefadj, timerefadj)
!referenceStateAdj(velDirFreestreamAdj,liftDirectionAdj,&
!      dragDirectionAdj, Machadj, MachCoefAdj,uInfAdj,prefAdj,&
!      rhorefAdj, pinfdimAdj, rhoinfdimAdj, rhoinfAdj, pinfAdj,&
!      murefAdj, timerefAdj)
!(velDirFreestreamAdj,liftDirectionAdj,&
!     dragDirectionAdj, Machadj, MachCoefAdj,uInfAdj)
  CALL SETFLOWINFINITYSTATEFORCESADJ(veldirfreestreamadj, &
&                               liftdirectionadj, dragdirectionadj, &
&                               machadj, machcoefadj, uinfadj, winfadj, &
&                               prefadj, rhorefadj, pinfdimadj, &
&                               rhoinfdimadj, rhoinfadj, pinfadj, &
&                               murefadj, timerefadj, pinfcorradj)
  CALL PUSHREAL8ARRAY(skadj, (iiend-iibeg+1)*(jjend-jjbeg+1)*2*3)
  CALL PUSHREAL8ARRAY(sjadj, (iiend-iibeg+1)*2*(jjend-jjbeg+1)*3)
  CALL PUSHREAL8ARRAY(siadj, 2*(iiend-iibeg+1)*(jjend-jjbeg+1)*3)
! Compute the surface normals (normAdj which is used only in 
! visous force computation) for the stencil
! Get siAdj,sjAdj,skAdj,normAdj
!      print *,'getting surface normals'
  CALL GETSURFACENORMALSADJ(xadj, siadj, sjadj, skadj, normadj, iibeg, &
&                      iiend, jjbeg, jjend, mm, level, nn, sps, &
&                      righthanded)
!     print *,'computing pressures'
  CALL COMPUTEFORCESPRESSUREADJ(wadj, padj)
  CALL PUSHREAL8ARRAY(padj, (ib+1)*(jb+1)*(kb+1))
  CALL PUSHREAL8ARRAY(wadj, (ib+1)*(jb+1)*(kb+1)*nw)
!    print *,'applyingbcs'
  CALL APPLYALLBCFORCESADJ(winfadj, pinfcorradj, wadj, padj, siadj, &
&                     sjadj, skadj, normadj, iibeg, iiend, jjbeg, jjend&
&                     , i2beg, i2end, j2beg, j2end, secondhalo, mm)
  CALL PUSHREAL8ARRAY(cmpadj, 3)
  CALL PUSHREAL8ARRAY(cfpadj, 3)
!   print *,'integrating forces'
! Integrate force components along the given subface
  CALL FORCESANDMOMENTSADJ(cfpadj, cmpadj, cfvadj, cmvadj, cfpadjout, &
&                     cmpadjout, cfvadjout, cmvadjout, yplusmax, &
&                     refpoint, siadj, sjadj, skadj, normadj, xadj, padj&
&                     , wadj, iibeg, iiend, jjbeg, jjend, i2beg, i2end, &
&                     j2beg, j2end, level, mm, nn, machcoefadj)
!(cFpAdj,cMpAdj, &
!     cFpAdjOut,cMpAdjOut, &
!     yplusMax,refPoint,siAdj,sjAdj,skAdj,normAdj,xAdj,pAdj,wAdj,&
!     iiBeg,iiEnd,jjBeg,jjEnd,i2Beg,i2End,j2Beg,j2End, &
!     level,mm,nn,machCoefAdj)
!end if invForce
! Compute the force components for the current block subface
  cmvadjoutb(:) = 0.0
  cmpadjoutb(:) = 0.0
  cmpadjoutb(3) = cmzadjb(sps)
  cmvadjoutb(3) = cmzadjb(sps)
  cmzadjb(sps) = 0.0
  cmpadjoutb(2) = cmpadjoutb(2) + cmyadjb(sps)
  cmvadjoutb(2) = cmvadjoutb(2) + cmyadjb(sps)
  cmyadjb(sps) = 0.0
  cmpadjoutb(1) = cmpadjoutb(1) + cmxadjb(sps)
  cmvadjoutb(1) = cmvadjoutb(1) + cmxadjb(sps)
  cmxadjb(sps) = 0.0
  cfpadjoutb(:) = 0.0
  dragdirectionadjb(:) = 0.0
  cfvadjoutb(:) = 0.0
  cfpadjoutb(1) = dragdirectionadj(1)*cdadjb(sps)
  cfvadjoutb(1) = dragdirectionadj(1)*cdadjb(sps)
  dragdirectionadjb(1) = (cfpadjout(1)+cfvadjout(1))*cdadjb(sps)
  cfpadjoutb(2) = dragdirectionadj(2)*cdadjb(sps)
  cfvadjoutb(2) = dragdirectionadj(2)*cdadjb(sps)
  dragdirectionadjb(2) = (cfpadjout(2)+cfvadjout(2))*cdadjb(sps)
  cfpadjoutb(3) = dragdirectionadj(3)*cdadjb(sps)
  cfvadjoutb(3) = dragdirectionadj(3)*cdadjb(sps)
  dragdirectionadjb(3) = (cfpadjout(3)+cfvadjout(3))*cdadjb(sps)
  cdadjb(sps) = 0.0
  liftdirectionadjb(:) = 0.0
  cfpadjoutb(1) = cfpadjoutb(1) + liftdirectionadj(1)*cladjb(sps)
  cfvadjoutb(1) = cfvadjoutb(1) + liftdirectionadj(1)*cladjb(sps)
  liftdirectionadjb(1) = (cfpadjout(1)+cfvadjout(1))*cladjb(sps)
  cfpadjoutb(2) = cfpadjoutb(2) + liftdirectionadj(2)*cladjb(sps)
  cfvadjoutb(2) = cfvadjoutb(2) + liftdirectionadj(2)*cladjb(sps)
  liftdirectionadjb(2) = (cfpadjout(2)+cfvadjout(2))*cladjb(sps)
  cfpadjoutb(3) = cfpadjoutb(3) + liftdirectionadj(3)*cladjb(sps)
  cfvadjoutb(3) = cfvadjoutb(3) + liftdirectionadj(3)*cladjb(sps)
  liftdirectionadjb(3) = (cfpadjout(3)+cfvadjout(3))*cladjb(sps)
  cladjb(sps) = 0.0
  CALL POPREAL8ARRAY(cfpadj, 3)
  CALL POPREAL8ARRAY(cmpadj, 3)
  CALL FORCESANDMOMENTSADJ_B(cfpadj, cmpadj, cfvadj, cmvadj, cfpadjout, &
&                       cfpadjoutb, cmpadjout, cmpadjoutb, cfvadjout, &
&                       cfvadjoutb, cmvadjout, cmvadjoutb, yplusmax, &
&                       refpoint, siadj, siadjb, sjadj, sjadjb, skadj, &
&                       skadjb, normadj, xadj, xadjb, padj, padjb, wadj&
&                       , iibeg, iiend, jjbeg, jjend, i2beg, i2end, &
&                       j2beg, j2end, level, mm, nn, machcoefadj, &
&                       machcoefadjb)
  CALL POPREAL8ARRAY(wadj, (ib+1)*(jb+1)*(kb+1)*nw)
  CALL POPREAL8ARRAY(padj, (ib+1)*(jb+1)*(kb+1))
  CALL APPLYALLBCFORCESADJ_B(winfadj, winfadjb, pinfcorradj, &
&                       pinfcorradjb, wadj, wadjb, padj, padjb, siadj, &
&                       siadjb, sjadj, sjadjb, skadj, skadjb, normadj, &
&                       normadjb, iibeg, iiend, jjbeg, jjend, i2beg, &
&                       i2end, j2beg, j2end, secondhalo, mm)
  CALL COMPUTEFORCESPRESSUREADJ_B(wadj, wadjb, padj, padjb)
  CALL POPREAL8ARRAY(siadj, 2*(iiend-iibeg+1)*(jjend-jjbeg+1)*3)
  CALL POPREAL8ARRAY(sjadj, (iiend-iibeg+1)*2*(jjend-jjbeg+1)*3)
  CALL POPREAL8ARRAY(skadj, (iiend-iibeg+1)*(jjend-jjbeg+1)*2*3)
  CALL GETSURFACENORMALSADJ_B(xadj, xadjb, siadj, siadjb, sjadj, sjadjb&
&                        , skadj, skadjb, normadj, normadjb, iibeg, &
&                        iiend, jjbeg, jjend, mm, level, nn, sps, &
&                        righthanded)
  CALL SETFLOWINFINITYSTATEFORCESADJ_B(veldirfreestreamadj, &
&                                 veldirfreestreamadjb, liftdirectionadj&
&                                 , dragdirectionadj, machadj, &
&                                 machcoefadj, uinfadj, uinfadjb, &
&                                 winfadj, winfadjb, prefadj, rhorefadj&
&                                 , pinfdimadj, rhoinfdimadj, rhoinfadj&
&                                 , pinfadj, murefadj, timerefadj, &
&                                 pinfcorradj, pinfcorradjb)
  CALL POPREAL8(prefadj)
  CALL POPREAL8(rhorefadj)
  CALL REFERENCESTATEFORCESADJ_B(machadj, machadjb, machcoefadj, uinfadj&
&                           , uinfadjb, prefadj, rhorefadj, pinfdimadj, &
&                           rhoinfdimadj, rhoinfadj, pinfadj, murefadj, &
&                           timerefadj)
  CALL POPREAL8ARRAY(veldirfreestreamadj, 3)
  CALL POPREAL8ARRAY(liftdirectionadj, 3)
  CALL POPREAL8(machcoefadj)
  CALL CHECKINPUTPARAMFORCESADJ_B(veldirfreestreamadj, &
&                            veldirfreestreamadjb, liftdirectionadj, &
&                            liftdirectionadjb, dragdirectionadj, &
&                            dragdirectionadjb, machadj, machadjb, &
&                            machcoefadj, machcoefadjb)
  CALL ADJUSTINFLOWANGLEFORCESADJ_B(alphaadj, alphaadjb, betaadj, &
&                              betaadjb, veldirfreestreamadj, &
&                              veldirfreestreamadjb, liftdirectionadj, &
&                              liftdirectionadjb, dragdirectionadj)
END SUBROUTINE COMPUTEFORCESADJ_B
