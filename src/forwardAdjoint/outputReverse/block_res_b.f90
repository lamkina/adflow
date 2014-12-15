   !        Generated by TAPENADE     (INRIA, Tropics team)
   !  Tapenade 3.10 (r5363) -  9 Sep 2014 09:53
   !
   !  Differentiation of block_res in reverse (adjoint) mode (with options i4 dr8 r8 noISIZE):
   !   gradient     of useful results: funcvalues *(flowdoms.x) *(flowdoms.w)
   !                *(flowdoms.dw) *(*bcdata.fp) *(*bcdata.fv) *(*bcdata.m)
   !                *(*bcdata.oarea) *(*bcdata.sepsensor) *(*bcdata.cavitation)
   !                moment force cavitation sepsensor
   !   with respect to varying inputs: funcvalues mach tempfreestream
   !                machgrid lengthref machcoef pointref *(flowdoms.x)
   !                *(flowdoms.w) *(flowdoms.dw) *(*bcdata.fp) *(*bcdata.fv)
   !                *(*bcdata.m) *(*bcdata.oarea) *(*bcdata.sepsensor)
   !                *(*bcdata.cavitation) pref moment alpha force
   !                beta cavitation sepsensor
   !   RW status of diff variables: funcvalues:in-zero mach:out tempfreestream:out
   !                veldirfreestream:(loc) machgrid:out lengthref:out
   !                machcoef:out dragdirection:(loc) liftdirection:(loc)
   !                pointref:out *(flowdoms.x):in-out *(flowdoms.vol):(loc)
   !                *(flowdoms.w):in-out *(flowdoms.dw):in-out *rev:(loc)
   !                *bvtj1:(loc) *bvtj2:(loc) *p:(loc) *gamma:(loc)
   !                *rlv:(loc) *bvtk1:(loc) *bvtk2:(loc) *si:(loc)
   !                *sj:(loc) *sk:(loc) *bvti1:(loc) *bvti2:(loc)
   !                *fw:(loc) *(*viscsubface.tau):(loc) *(*bcdata.norm):(loc)
   !                *(*bcdata.fp):in-out *(*bcdata.fv):in-out *(*bcdata.m):in-out
   !                *(*bcdata.oarea):in-out *(*bcdata.sepsensor):in-out
   !                *(*bcdata.cavitation):in-out *radi:(loc) *radj:(loc)
   !                *radk:(loc) mudim:(loc) gammainf:(loc) pinf:(loc)
   !                timeref:(loc) rhoinf:(loc) muref:(loc) rhoinfdim:(loc)
   !                tref:(loc) winf:(loc) muinf:(loc) uinf:(loc) pinfcorr:(loc)
   !                rgas:(loc) pinfdim:(loc) pref:out rhoref:(loc)
   !                moment:in-zero alpha:out force:in-zero beta:out
   !                cavitation:in-zero sepsensor:in-zero
   !   Plus diff mem management of: flowdoms.x:in flowdoms.vol:in
   !                flowdoms.w:in flowdoms.dw:in rev:in bvtj1:in bvtj2:in
   !                p:in gamma:in rlv:in bvtk1:in bvtk2:in si:in sj:in
   !                sk:in bvti1:in bvti2:in fw:in viscsubface:in *viscsubface.tau:in
   !                bcdata:in *bcdata.norm:in *bcdata.fp:in *bcdata.fv:in
   !                *bcdata.m:in *bcdata.oarea:in *bcdata.sepsensor:in
   !                *bcdata.cavitation:in radi:in radj:in radk:in
   ! This is a super-combined function that combines the original
   ! functionality of: 
   ! Pressure Computation
   ! timeStep
   ! applyAllBCs
   ! initRes
   ! residual 
   ! The real difference between this and the original modules is that it
   ! it only operates on a single block at a time and as such the nominal
   ! block/sps loop is outside the calculation. This routine is suitable
   ! for forward mode AD with Tapenade
   SUBROUTINE BLOCK_RES_B(nn, sps, usespatial, alpha, alphab, beta, betab, &
   & liftindex, force, forceb, moment, momentb, sepsensor, sepsensorb, &
   & cavitation, cavitationb)
   USE BLOCKPOINTERS_B
   USE FLOWVARREFSTATE
   USE INPUTPHYSICS
   USE INPUTITERATION
   USE INPUTTIMESPECTRAL
   USE SECTION
   USE MONITOR
   USE ITERATION
   USE INPUTADJOINT
   USE DIFFSIZES
   USE COSTFUNCTIONS
   IMPLICIT NONE
   ! Input Arguments:
   INTEGER(kind=inttype), INTENT(IN) :: nn, sps
   LOGICAL, INTENT(IN) :: usespatial
   REAL(kind=realtype), INTENT(IN) :: alpha, beta
   REAL(kind=realtype) :: alphab, betab
   INTEGER(kind=inttype), INTENT(IN) :: liftindex
   ! Output Variables
   REAL(kind=realtype), DIMENSION(3, ntimeintervalsspectral) :: force, &
   & moment
   REAL(kind=realtype), DIMENSION(3, ntimeintervalsspectral) :: forceb, &
   & momentb
   REAL(kind=realtype) :: sepsensor, cavitation
   REAL(kind=realtype) :: sepsensorb, cavitationb
   ! Working Variables
   REAL(kind=realtype) :: gm1, v2, fact, tmp
   REAL(kind=realtype) :: v2b, factb, tmpb
   INTEGER(kind=inttype) :: i, j, k, sps2, mm, l, ii, ll, jj
   INTEGER(kind=inttype) :: nstate
   REAL(kind=realtype), DIMENSION(nsections) :: t
   LOGICAL :: useoldcoor
   REAL(kind=realtype), DIMENSION(3) :: cfp, cfv, cmp, cmv
   REAL(kind=realtype), DIMENSION(3) :: cfpb, cfvb, cmpb, cmvb
   REAL(kind=realtype) :: yplusmax, scaledim
   REAL(kind=realtype) :: scaledimb
   INTRINSIC MAX
   INTEGER :: branch
   REAL(kind=realtype) :: temp3
   REAL(kind=realtype) :: temp2
   REAL(kind=realtype) :: temp1
   REAL(kind=realtype) :: temp0
   REAL(kind=realtype) :: tempb7(3)
   REAL(kind=realtype) :: tempb6
   REAL(kind=realtype) :: tempb5(3)
   REAL(kind=realtype) :: tempb4
   REAL(kind=realtype) :: tempb3
   REAL(kind=realtype) :: tempb2
   REAL(kind=realtype) :: tempb1
   REAL(kind=realtype) :: tempb0
   REAL(kind=realtype) :: tempb
   INTEGER :: ii3
   INTEGER :: ii2
   INTEGER :: ii1
   REAL(kind=realtype) :: temp
   REAL(kind=realtype) :: temp4
   ! Setup number of state variable based on turbulence assumption
   IF (frozenturbulence) THEN
   nstate = nwf
   ELSE
   nstate = nw
   END IF
   ! Set pointers to input/output variables
   wb => flowdomsb(nn, currentlevel, sps)%w
   w => flowdoms(nn, currentlevel, sps)%w
   dwb => flowdomsb(nn, 1, sps)%dw
   dw => flowdoms(nn, 1, sps)%dw
   xb => flowdomsb(nn, currentlevel, sps)%x
   x => flowdoms(nn, currentlevel, sps)%x
   volb => flowdomsb(nn, currentlevel, sps)%vol
   vol => flowdoms(nn, currentlevel, sps)%vol
   ! ------------------------------------------------
   !        Additional 'Extra' Components
   ! ------------------------------------------------ 
   CALL ADJUSTINFLOWANGLE(alpha, beta, liftindex)
   CALL PUSHREAL8(rhoref)
   CALL PUSHREAL8(pref)
   CALL PUSHREAL8(tref)
   CALL PUSHREAL8(gammainf)
   CALL REFERENCESTATE()
   CALL SETFLOWINFINITYSTATE()
   ! ------------------------------------------------
   !        Additional Spatial Components
   ! ------------------------------------------------
   IF (usespatial) THEN
   CALL XHALO_BLOCK()
   CALL PUSHREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk&
   &                 , 4))
   CALL PUSHREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj&
   &                 , 4))
   CALL PUSHREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si&
   &                 , 4))
   CALL METRIC_BLOCK()
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   ! ------------------------------------------------
   !        Normal Residual Computation
   ! ------------------------------------------------
   ! Compute the pressures
   gm1 = gammaconstant - one
   ! Compute P 
   DO k=0,kb
   DO j=0,jb
   DO i=0,ib
   CALL PUSHREAL8(v2)
   v2 = w(i, j, k, ivx)**2 + w(i, j, k, ivy)**2 + w(i, j, k, ivz)**&
   &         2
   p(i, j, k) = gm1*(w(i, j, k, irhoe)-half*w(i, j, k, irho)*v2)
   IF (p(i, j, k) .LT. 1.e-4_realType*pinfcorr) THEN
   p(i, j, k) = 1.e-4_realType*pinfcorr
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   p(i, j, k) = p(i, j, k)
   END IF
   END DO
   END DO
   END DO
   ! Compute Laminar/eddy viscosity if required
   CALL PUSHREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL COMPUTELAMVISCOSITY()
   CALL COMPUTEEDDYVISCOSITY()
   !  Apply all BC's
   CALL PUSHREAL8ARRAY(bmtj2, SIZE(bmtj2, 1)*SIZE(bmtj2, 2)*SIZE(bmtj2, 3&
   &               )*SIZE(bmtj2, 4))
   CALL PUSHREAL8ARRAY(bmtj1, SIZE(bmtj1, 1)*SIZE(bmtj1, 2)*SIZE(bmtj1, 3&
   &               )*SIZE(bmtj1, 4))
   CALL PUSHREAL8ARRAY(bvti2, SIZE(bvti2, 1)*SIZE(bvti2, 2)*SIZE(bvti2, 3&
   &               ))
   CALL PUSHREAL8ARRAY(bvti1, SIZE(bvti1, 1)*SIZE(bvti1, 2)*SIZE(bvti1, 3&
   &               ))
   CALL PUSHREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk, 4&
   &               ))
   CALL PUSHREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj, 4&
   &               ))
   CALL PUSHREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si, 4&
   &               ))
   CALL PUSHREAL8ARRAY(bmti2, SIZE(bmti2, 1)*SIZE(bmti2, 2)*SIZE(bmti2, 3&
   &               )*SIZE(bmti2, 4))
   CALL PUSHREAL8ARRAY(bmti1, SIZE(bmti1, 1)*SIZE(bmti1, 2)*SIZE(bmti1, 3&
   &               )*SIZE(bmti1, 4))
   CALL PUSHREAL8ARRAY(bvtk2, SIZE(bvtk2, 1)*SIZE(bvtk2, 2)*SIZE(bvtk2, 3&
   &               ))
   CALL PUSHREAL8ARRAY(bvtk1, SIZE(bvtk1, 1)*SIZE(bvtk1, 2)*SIZE(bvtk1, 3&
   &               ))
   CALL PUSHREAL8ARRAY(rlv, SIZE(rlv, 1)*SIZE(rlv, 2)*SIZE(rlv, 3))
   CALL PUSHREAL8ARRAY(bmtk2, SIZE(bmtk2, 1)*SIZE(bmtk2, 2)*SIZE(bmtk2, 3&
   &               )*SIZE(bmtk2, 4))
   CALL PUSHREAL8ARRAY(bmtk1, SIZE(bmtk1, 1)*SIZE(bmtk1, 2)*SIZE(bmtk1, 3&
   &               )*SIZE(bmtk1, 4))
   CALL PUSHREAL8ARRAY(gamma, SIZE(gamma, 1)*SIZE(gamma, 2)*SIZE(gamma, 3&
   &               ))
   CALL PUSHREAL8ARRAY(s, SIZE(s, 1)*SIZE(s, 2)*SIZE(s, 3)*SIZE(s, 4))
   CALL PUSHREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL PUSHREAL8ARRAY(bvtj2, SIZE(bvtj2, 1)*SIZE(bvtj2, 2)*SIZE(bvtj2, 3&
   &               ))
   CALL PUSHREAL8ARRAY(bvtj1, SIZE(bvtj1, 1)*SIZE(bvtj1, 2)*SIZE(bvtj1, 3&
   &               ))
   CALL PUSHREAL8ARRAY(rev, SIZE(rev, 1)*SIZE(rev, 2)*SIZE(rev, 3))
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                     , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, &
   &                     2)*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(&
   &                     flowdoms(ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   CALL APPLYALLBC_BLOCK(.true.)
   ! if (equations == RANSequations) then 
   !    call applyAllTurbBCThisBLock(.True.)
   ! end if
   ! Compute skin_friction Velocity (only for wall Functions)
   ! #ifndef 1
   !   call computeUtau_block
   ! #endif
   ! Compute time step and spectral radius
   CALL PUSHREAL8ARRAY(radk, SIZE(radk, 1)*SIZE(radk, 2)*SIZE(radk, 3))
   CALL PUSHREAL8ARRAY(radj, SIZE(radj, 1)*SIZE(radj, 2)*SIZE(radj, 3))
   CALL PUSHREAL8ARRAY(radi, SIZE(radi, 1)*SIZE(radi, 2)*SIZE(radi, 3))
   CALL TIMESTEP_BLOCK(.false.)
   spectralloop0:DO sps2=1,ntimeintervalsspectral
   flowdoms(nn, 1, sps2)%dw(:, :, :, :) = zero
   END DO spectralloop0
   ! -------------------------------
   ! Compute turbulence residual for RANS equations
   IF (equations .EQ. ransequations) THEN
   ! ! Initialize only the Turblent Variables
   ! call unsteadyTurbSpectral_block(itu1, itu1, nn, sps)
   SELECT CASE  (turbmodel) 
   CASE (spalartallmaras) 
   !call determineDistance2(1, sps)
   CALL PUSHREAL8ARRAY(bmtj2, SIZE(bmtj2, 1)*SIZE(bmtj2, 2)*SIZE(&
   &                   bmtj2, 3)*SIZE(bmtj2, 4))
   CALL PUSHREAL8ARRAY(bmtj1, SIZE(bmtj1, 1)*SIZE(bmtj1, 2)*SIZE(&
   &                   bmtj1, 3)*SIZE(bmtj1, 4))
   CALL PUSHREAL8ARRAY(bmti2, SIZE(bmti2, 1)*SIZE(bmti2, 2)*SIZE(&
   &                   bmti2, 3)*SIZE(bmti2, 4))
   CALL PUSHREAL8ARRAY(bmti1, SIZE(bmti1, 1)*SIZE(bmti1, 2)*SIZE(&
   &                   bmti1, 3)*SIZE(bmti1, 4))
   CALL PUSHREAL8ARRAY(bmtk2, SIZE(bmtk2, 1)*SIZE(bmtk2, 2)*SIZE(&
   &                   bmtk2, 3)*SIZE(bmtk2, 4))
   CALL PUSHREAL8ARRAY(bmtk1, SIZE(bmtk1, 1)*SIZE(bmtk1, 2)*SIZE(&
   &                   bmtk1, 3)*SIZE(bmtk1, 4))
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%dw, SIZE(&
   &                         flowdoms(ii3, ii2, ii1)%dw, 1)*SIZE(flowdoms(&
   &                         ii3, ii2, ii1)%dw, 2)*SIZE(flowdoms(ii3, ii2, &
   &                         ii1)%dw, 3)*SIZE(flowdoms(ii3, ii2, ii1)%dw, 4&
   &                         ))
   END DO
   END DO
   END DO
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms&
   &                         (ii3, ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, &
   &                         ii1)%w, 2)*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*&
   &                         SIZE(flowdoms(ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   CALL SA_BLOCK(.true.)
   CALL PUSHCONTROL2B(0)
   CASE DEFAULT
   CALL PUSHCONTROL2B(1)
   END SELECT
   ELSE
   CALL PUSHCONTROL2B(2)
   END IF
   ! -------------------------------  
   ! Next initialize residual for flow variables. The is the only place
   ! where there is an n^2 dependance. There are issues with
   ! initRes. So only the necesary timespectral code has been copied
   ! here. See initres for more information and comments.
   ! sps here is the on-spectral instance
   IF (ntimeintervalsspectral .EQ. 1) THEN
   dw(:, :, :, 1:nwf) = zero
   CALL PUSHCONTROL1B(0)
   ELSE
   ! Zero dw on all spectral instances
   spectralloop1:DO sps2=1,ntimeintervalsspectral
   flowdoms(nn, 1, sps2)%dw(:, :, :, 1:nwf) = zero
   END DO spectralloop1
   spectralloop2:DO sps2=1,ntimeintervalsspectral
   CALL PUSHINTEGER4(jj)
   jj = sectionid
   timeloopfine:DO mm=1,ntimeintervalsspectral
   CALL PUSHINTEGER4(ii)
   ii = 3*(mm-1)
   varloopfine:DO l=1,nwf
   IF ((l .EQ. ivx .OR. l .EQ. ivy) .OR. l .EQ. ivz) THEN
   IF (l .EQ. ivx) THEN
   CALL PUSHINTEGER4(ll)
   ll = 3*sps2 - 2
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (l .EQ. ivy) THEN
   CALL PUSHINTEGER4(ll)
   ll = 3*sps2 - 1
   CALL PUSHCONTROL1B(0)
   ELSE
   CALL PUSHCONTROL1B(1)
   END IF
   IF (l .EQ. ivz) THEN
   CALL PUSHINTEGER4(ll)
   ll = 3*sps2
   CALL PUSHCONTROL1B(1)
   ELSE
   CALL PUSHCONTROL1B(0)
   END IF
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   CALL PUSHREAL8(tmp)
   tmp = dvector(jj, ll, ii+1)*flowdoms(nn, 1, mm)%w(i, j&
   &                   , k, ivx) + dvector(jj, ll, ii+2)*flowdoms(nn, 1, mm&
   &                   )%w(i, j, k, ivy) + dvector(jj, ll, ii+3)*flowdoms(&
   &                   nn, 1, mm)%w(i, j, k, ivz)
   flowdoms(nn, 1, sps2)%dw(i, j, k, l) = flowdoms(nn, 1&
   &                   , sps2)%dw(i, j, k, l) + tmp*flowdoms(nn, 1, mm)%vol&
   &                   (i, j, k)*flowdoms(nn, 1, mm)%w(i, j, k, irho)
   END DO
   END DO
   END DO
   CALL PUSHCONTROL1B(1)
   ELSE
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   ! This is: dw = dw + dscalar*vol*w
   flowdoms(nn, 1, sps2)%dw(i, j, k, l) = flowdoms(nn, 1&
   &                   , sps2)%dw(i, j, k, l) + dscalar(jj, sps2, mm)*&
   &                   flowdoms(nn, 1, mm)%vol(i, j, k)*flowdoms(nn, 1, mm)&
   &                   %w(i, j, k, l)
   END DO
   END DO
   END DO
   CALL PUSHCONTROL1B(0)
   END IF
   END DO varloopfine
   END DO timeloopfine
   END DO spectralloop2
   CALL PUSHCONTROL1B(1)
   END IF
   !  Actual residual calc
   CALL PUSHREAL8ARRAY(fw, SIZE(fw, 1)*SIZE(fw, 2)*SIZE(fw, 3)*SIZE(fw, 4&
   &               ))
   CALL PUSHREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%dw, SIZE(flowdoms(&
   &                     ii3, ii2, ii1)%dw, 1)*SIZE(flowdoms(ii3, ii2, ii1)&
   &                     %dw, 2)*SIZE(flowdoms(ii3, ii2, ii1)%dw, 3)*SIZE(&
   &                     flowdoms(ii3, ii2, ii1)%dw, 4))
   END DO
   END DO
   END DO
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                     , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, &
   &                     2)*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(&
   &                     flowdoms(ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   CALL RESIDUAL_BLOCK()
   ! Note that there are some error introduced by viscousflux from fw
   ! The error only show up in the rho term in some cells
   ! Divide through by the volume
   DO sps2=1,ntimeintervalsspectral
   DO l=1,nwf
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   CALL PUSHREAL8(flowdoms(nn, 1, sps2)%dw(i, j, k, l))
   flowdoms(nn, 1, sps2)%dw(i, j, k, l) = flowdoms(nn, 1, sps2)&
   &             %dw(i, j, k, l)/flowdoms(nn, currentlevel, sps2)%vol(i, j&
   &             , k)
   END DO
   END DO
   END DO
   END DO
   ! Treat the turblent residual with the scaling factor on the
   ! residual
   DO l=nt1,nstate
   DO k=2,kl
   DO j=2,jl
   DO i=2,il
   CALL PUSHREAL8(flowdoms(nn, 1, sps2)%dw(i, j, k, l))
   flowdoms(nn, 1, sps2)%dw(i, j, k, l) = flowdoms(nn, 1, sps2)&
   &             %dw(i, j, k, l)/flowdoms(nn, currentlevel, sps2)%vol(i, j&
   &             , k)*turbresscale
   END DO
   END DO
   END DO
   END DO
   END DO
   DO ii1=1,SIZE(bcdata)
   CALL PUSHREAL8ARRAY(bcdata(ii1)%oarea, SIZE(bcdata(ii1)%oarea, 1)*&
   &                 SIZE(bcdata(ii1)%oarea, 2))
   END DO
   CALL PUSHREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk, 4&
   &               ))
   CALL PUSHREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj, 4&
   &               ))
   CALL PUSHREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si, 4&
   &               ))
   CALL PUSHREAL8ARRAY(d2wall, SIZE(d2wall, 1)*SIZE(d2wall, 2)*SIZE(&
   &               d2wall, 3))
   CALL PUSHREAL8ARRAY(rlv, SIZE(rlv, 1)*SIZE(rlv, 2)*SIZE(rlv, 3))
   CALL PUSHREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL PUSHREAL8ARRAY(rev, SIZE(rev, 1)*SIZE(rev, 2)*SIZE(rev, 3))
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                     , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, &
   &                     2)*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(&
   &                     flowdoms(ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   CALL PUSHREAL8ARRAY(flowdoms(ii3, ii2, ii1)%x, SIZE(flowdoms(ii3&
   &                     , ii2, ii1)%x, 1)*SIZE(flowdoms(ii3, ii2, ii1)%x, &
   &                     2)*SIZE(flowdoms(ii3, ii2, ii1)%x, 3)*SIZE(&
   &                     flowdoms(ii3, ii2, ii1)%x, 4))
   END DO
   END DO
   END DO
   CALL PUSHREAL8ARRAY(cmv, 3)
   CALL PUSHREAL8ARRAY(cmp, 3)
   CALL PUSHREAL8ARRAY(cfv, 3)
   CALL PUSHREAL8ARRAY(cfp, 3)
   CALL FORCESANDMOMENTS(cfp, cfv, cmp, cmv, yplusmax, sepsensor, &
   &                    cavitation)
   ! Convert back to actual forces. Note that even though we use
   ! MachCoef, Lref, and surfaceRef here, they are NOT differented,
   ! since F doesn't actually depend on them. Ideally we would just get
   ! the raw forces and moment form forcesAndMoments. 
   force = zero
   moment = zero
   scaledim = pref/pinf
   fact = two/(gammainf*pinf*machcoef*machcoef*surfaceref*lref*lref*&
   &   scaledim)
   DO sps2=1,ntimeintervalsspectral
   force(:, sps2) = (cfp+cfv)/fact
   END DO
   CALL PUSHREAL8(fact)
   fact = fact/(lengthref*lref)
   DO sps2=1,ntimeintervalsspectral
   moment(:, sps2) = (cmp+cmv)/fact
   END DO
   CALL GETCOSTFUNCTION2_B(force, forceb, moment, momentb, sepsensor, &
   &                   sepsensorb, cavitation, cavitationb, alpha, beta, &
   &                   liftindex)
   cmpb = 0.0_8
   cmvb = 0.0_8
   factb = 0.0_8
   DO sps2=ntimeintervalsspectral,1,-1
   tempb7 = momentb(:, sps2)/fact
   cmpb = cmpb + tempb7
   cmvb = cmvb + tempb7
   factb = factb + SUM(-((cmp+cmv)*tempb7/fact))
   momentb(:, sps2) = 0.0_8
   END DO
   CALL POPREAL8(fact)
   tempb6 = factb/(lref*lengthref)
   lengthrefb = lengthrefb - fact*tempb6/lengthref
   factb = tempb6
   cfpb = 0.0_8
   cfvb = 0.0_8
   DO sps2=ntimeintervalsspectral,1,-1
   tempb5 = forceb(:, sps2)/fact
   cfpb = cfpb + tempb5
   cfvb = cfvb + tempb5
   factb = factb + SUM(-((cfp+cfv)*tempb5/fact))
   forceb(:, sps2) = 0.0_8
   END DO
   temp4 = machcoef**2*scaledim
   temp3 = surfaceref*lref**2
   temp2 = temp3*gammainf*pinf
   tempb3 = -(two*factb/(temp2**2*temp4**2))
   tempb4 = temp4*temp3*tempb3
   gammainfb = gammainfb + pinf*tempb4
   machcoefb = machcoefb + scaledim*temp2*2*machcoef*tempb3
   scaledimb = temp2*machcoef**2*tempb3
   pinfb = pinfb + gammainf*tempb4 - pref*scaledimb/pinf**2
   prefb = prefb + scaledimb/pinf
   CALL POPREAL8ARRAY(cfp, 3)
   CALL POPREAL8ARRAY(cfv, 3)
   CALL POPREAL8ARRAY(cmp, 3)
   CALL POPREAL8ARRAY(cmv, 3)
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%x, SIZE(flowdoms(ii3&
   &                    , ii2, ii1)%x, 1)*SIZE(flowdoms(ii3, ii2, ii1)%x, 2&
   &                    )*SIZE(flowdoms(ii3, ii2, ii1)%x, 3)*SIZE(flowdoms(&
   &                    ii3, ii2, ii1)%x, 4))
   END DO
   END DO
   END DO
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                    , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, 2&
   &                    )*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(flowdoms(&
   &                    ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   CALL POPREAL8ARRAY(rev, SIZE(rev, 1)*SIZE(rev, 2)*SIZE(rev, 3))
   CALL POPREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL POPREAL8ARRAY(rlv, SIZE(rlv, 1)*SIZE(rlv, 2)*SIZE(rlv, 3))
   CALL POPREAL8ARRAY(d2wall, SIZE(d2wall, 1)*SIZE(d2wall, 2)*SIZE(d2wall&
   &              , 3))
   CALL POPREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si, 4)&
   &             )
   CALL POPREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj, 4)&
   &             )
   CALL POPREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk, 4)&
   &             )
    DO ii1=SIZE(bcdata),1,-1
   CALL POPREAL8ARRAY(bcdata(ii1)%oarea, SIZE(bcdata(ii1)%oarea, 1)*&
   &                SIZE(bcdata(ii1)%oarea, 2))
   END DO
   CALL FORCESANDMOMENTS_B(cfp, cfpb, cfv, cfvb, cmp, cmpb, cmv, cmvb, &
   &                   yplusmax, sepsensor, sepsensorb, cavitation, &
   &                   cavitationb)
   DO ii1=1,ntimeintervalsspectral
   DO ii2=1,1
   DO ii3=nn,nn
   flowdomsb(ii3, ii2, ii1)%vol = 0.0_8
   END DO
   END DO
   END DO
   DO sps2=ntimeintervalsspectral,1,-1
   DO l=nstate,nt1,-1
   DO k=kl,2,-1
   DO j=jl,2,-1
   DO i=il,2,-1
   CALL POPREAL8(flowdoms(nn, 1, sps2)%dw(i, j, k, l))
   temp1 = flowdoms(nn, currentlevel, sps2)%vol(i, j, k)
   tempb2 = turbresscale*flowdomsb(nn, 1, sps2)%dw(i, j, k, l)/&
   &             temp1
   flowdomsb(nn, currentlevel, sps2)%vol(i, j, k) = flowdomsb(&
   &             nn, currentlevel, sps2)%vol(i, j, k) - flowdoms(nn, 1, &
   &             sps2)%dw(i, j, k, l)*tempb2/temp1
   flowdomsb(nn, 1, sps2)%dw(i, j, k, l) = tempb2
   END DO
   END DO
   END DO
   END DO
   DO l=nwf,1,-1
   DO k=kl,2,-1
   DO j=jl,2,-1
   DO i=il,2,-1
   CALL POPREAL8(flowdoms(nn, 1, sps2)%dw(i, j, k, l))
   temp0 = flowdoms(nn, currentlevel, sps2)%vol(i, j, k)
   flowdomsb(nn, currentlevel, sps2)%vol(i, j, k) = flowdomsb(&
   &             nn, currentlevel, sps2)%vol(i, j, k) - flowdoms(nn, 1, &
   &             sps2)%dw(i, j, k, l)*flowdomsb(nn, 1, sps2)%dw(i, j, k, l)&
   &             /temp0**2
   flowdomsb(nn, 1, sps2)%dw(i, j, k, l) = flowdomsb(nn, 1, &
   &             sps2)%dw(i, j, k, l)/temp0
   END DO
   END DO
   END DO
   END DO
   END DO
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                    , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, 2&
   &                    )*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(flowdoms(&
   &                    ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%dw, SIZE(flowdoms(ii3&
   &                    , ii2, ii1)%dw, 1)*SIZE(flowdoms(ii3, ii2, ii1)%dw&
   &                    , 2)*SIZE(flowdoms(ii3, ii2, ii1)%dw, 3)*SIZE(&
   &                    flowdoms(ii3, ii2, ii1)%dw, 4))
   END DO
   END DO
   END DO
   CALL POPREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL POPREAL8ARRAY(fw, SIZE(fw, 1)*SIZE(fw, 2)*SIZE(fw, 3)*SIZE(fw, 4)&
   &             )
   CALL RESIDUAL_BLOCK_B()
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   dwb(:, :, :, 1:nwf) = 0.0_8
   ELSE
   DO sps2=ntimeintervalsspectral,1,-1
   DO mm=ntimeintervalsspectral,1,-1
   DO l=nwf,1,-1
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   DO k=kl,2,-1
   DO j=jl,2,-1
   DO i=il,2,-1
   tempb1 = dscalar(jj, sps2, mm)*flowdomsb(nn, 1, sps2)%&
   &                   dw(i, j, k, l)
   flowdomsb(nn, 1, mm)%vol(i, j, k) = flowdomsb(nn, 1, &
   &                   mm)%vol(i, j, k) + flowdoms(nn, 1, mm)%w(i, j, k, l)&
   &                   *tempb1
   flowdomsb(nn, 1, mm)%w(i, j, k, l) = flowdomsb(nn, 1, &
   &                   mm)%w(i, j, k, l) + flowdoms(nn, 1, mm)%vol(i, j, k)&
   &                   *tempb1
   END DO
   END DO
   END DO
   ELSE
   DO k=kl,2,-1
   DO j=jl,2,-1
   DO i=il,2,-1
   tempb0 = flowdoms(nn, 1, mm)%w(i, j, k, irho)*&
   &                   flowdomsb(nn, 1, sps2)%dw(i, j, k, l)
   temp = flowdoms(nn, 1, mm)%vol(i, j, k)
   tmpb = temp*tempb0
   flowdomsb(nn, 1, mm)%vol(i, j, k) = flowdomsb(nn, 1, &
   &                   mm)%vol(i, j, k) + tmp*tempb0
   flowdomsb(nn, 1, mm)%w(i, j, k, irho) = flowdomsb(nn, &
   &                   1, mm)%w(i, j, k, irho) + tmp*temp*flowdomsb(nn, 1, &
   &                   sps2)%dw(i, j, k, l)
   CALL POPREAL8(tmp)
   flowdomsb(nn, 1, mm)%w(i, j, k, ivx) = flowdomsb(nn, 1&
   &                   , mm)%w(i, j, k, ivx) + dvector(jj, ll, ii+1)*tmpb
   flowdomsb(nn, 1, mm)%w(i, j, k, ivy) = flowdomsb(nn, 1&
   &                   , mm)%w(i, j, k, ivy) + dvector(jj, ll, ii+2)*tmpb
   flowdomsb(nn, 1, mm)%w(i, j, k, ivz) = flowdomsb(nn, 1&
   &                   , mm)%w(i, j, k, ivz) + dvector(jj, ll, ii+3)*tmpb
   END DO
   END DO
   END DO
   CALL POPCONTROL1B(branch)
   IF (branch .NE. 0) CALL POPINTEGER4(ll)
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) CALL POPINTEGER4(ll)
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) CALL POPINTEGER4(ll)
   END IF
   END DO
   CALL POPINTEGER4(ii)
   END DO
   CALL POPINTEGER4(jj)
   END DO
   DO sps2=ntimeintervalsspectral,1,-1
   flowdomsb(nn, 1, sps2)%dw(:, :, :, 1:nwf) = 0.0_8
   END DO
   END IF
   CALL POPCONTROL2B(branch)
   IF (branch .EQ. 0) THEN
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(&
   &                      ii3, ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)&
   &                      %w, 2)*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(&
   &                      flowdoms(ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%dw, SIZE(flowdoms(&
   &                      ii3, ii2, ii1)%dw, 1)*SIZE(flowdoms(ii3, ii2, ii1&
   &                      )%dw, 2)*SIZE(flowdoms(ii3, ii2, ii1)%dw, 3)*SIZE&
   &                      (flowdoms(ii3, ii2, ii1)%dw, 4))
   END DO
   END DO
   END DO
   CALL POPREAL8ARRAY(bmtk1, SIZE(bmtk1, 1)*SIZE(bmtk1, 2)*SIZE(bmtk1, &
   &                3)*SIZE(bmtk1, 4))
   CALL POPREAL8ARRAY(bmtk2, SIZE(bmtk2, 1)*SIZE(bmtk2, 2)*SIZE(bmtk2, &
   &                3)*SIZE(bmtk2, 4))
   CALL POPREAL8ARRAY(bmti1, SIZE(bmti1, 1)*SIZE(bmti1, 2)*SIZE(bmti1, &
   &                3)*SIZE(bmti1, 4))
   CALL POPREAL8ARRAY(bmti2, SIZE(bmti2, 1)*SIZE(bmti2, 2)*SIZE(bmti2, &
   &                3)*SIZE(bmti2, 4))
   CALL POPREAL8ARRAY(bmtj1, SIZE(bmtj1, 1)*SIZE(bmtj1, 2)*SIZE(bmtj1, &
   &                3)*SIZE(bmtj1, 4))
   CALL POPREAL8ARRAY(bmtj2, SIZE(bmtj2, 1)*SIZE(bmtj2, 2)*SIZE(bmtj2, &
   &                3)*SIZE(bmtj2, 4))
   CALL SA_BLOCK_B(.true.)
   ELSE IF (branch .EQ. 1) THEN
   bvtj1b = 0.0_8
   bvtj2b = 0.0_8
   bvtk1b = 0.0_8
   bvtk2b = 0.0_8
   bvti1b = 0.0_8
   bvti2b = 0.0_8
   ELSE
   bvtj1b = 0.0_8
   bvtj2b = 0.0_8
   bvtk1b = 0.0_8
   bvtk2b = 0.0_8
   bvti1b = 0.0_8
   bvti2b = 0.0_8
   END IF
   DO sps2=ntimeintervalsspectral,1,-1
   flowdomsb(nn, 1, sps2)%dw = 0.0_8
   END DO
   CALL POPREAL8ARRAY(radi, SIZE(radi, 1)*SIZE(radi, 2)*SIZE(radi, 3))
   CALL POPREAL8ARRAY(radj, SIZE(radj, 1)*SIZE(radj, 2)*SIZE(radj, 3))
   CALL POPREAL8ARRAY(radk, SIZE(radk, 1)*SIZE(radk, 2)*SIZE(radk, 3))
   CALL TIMESTEP_BLOCK_B(.false.)
   DO ii1=ntimeintervalsspectral,1,-1
   DO ii2=1,1,-1
   DO ii3=nn,nn,-1
   CALL POPREAL8ARRAY(flowdoms(ii3, ii2, ii1)%w, SIZE(flowdoms(ii3&
   &                    , ii2, ii1)%w, 1)*SIZE(flowdoms(ii3, ii2, ii1)%w, 2&
   &                    )*SIZE(flowdoms(ii3, ii2, ii1)%w, 3)*SIZE(flowdoms(&
   &                    ii3, ii2, ii1)%w, 4))
   END DO
   END DO
   END DO
   CALL POPREAL8ARRAY(rev, SIZE(rev, 1)*SIZE(rev, 2)*SIZE(rev, 3))
   CALL POPREAL8ARRAY(bvtj1, SIZE(bvtj1, 1)*SIZE(bvtj1, 2)*SIZE(bvtj1, 3)&
   &             )
   CALL POPREAL8ARRAY(bvtj2, SIZE(bvtj2, 1)*SIZE(bvtj2, 2)*SIZE(bvtj2, 3)&
   &             )
   CALL POPREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL POPREAL8ARRAY(s, SIZE(s, 1)*SIZE(s, 2)*SIZE(s, 3)*SIZE(s, 4))
   CALL POPREAL8ARRAY(gamma, SIZE(gamma, 1)*SIZE(gamma, 2)*SIZE(gamma, 3)&
   &             )
   CALL POPREAL8ARRAY(bmtk1, SIZE(bmtk1, 1)*SIZE(bmtk1, 2)*SIZE(bmtk1, 3)&
   &              *SIZE(bmtk1, 4))
   CALL POPREAL8ARRAY(bmtk2, SIZE(bmtk2, 1)*SIZE(bmtk2, 2)*SIZE(bmtk2, 3)&
   &              *SIZE(bmtk2, 4))
   CALL POPREAL8ARRAY(rlv, SIZE(rlv, 1)*SIZE(rlv, 2)*SIZE(rlv, 3))
   CALL POPREAL8ARRAY(bvtk1, SIZE(bvtk1, 1)*SIZE(bvtk1, 2)*SIZE(bvtk1, 3)&
   &             )
   CALL POPREAL8ARRAY(bvtk2, SIZE(bvtk2, 1)*SIZE(bvtk2, 2)*SIZE(bvtk2, 3)&
   &             )
   CALL POPREAL8ARRAY(bmti1, SIZE(bmti1, 1)*SIZE(bmti1, 2)*SIZE(bmti1, 3)&
   &              *SIZE(bmti1, 4))
   CALL POPREAL8ARRAY(bmti2, SIZE(bmti2, 1)*SIZE(bmti2, 2)*SIZE(bmti2, 3)&
   &              *SIZE(bmti2, 4))
   CALL POPREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si, 4)&
   &             )
   CALL POPREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj, 4)&
   &             )
   CALL POPREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk, 4)&
   &             )
   CALL POPREAL8ARRAY(bvti1, SIZE(bvti1, 1)*SIZE(bvti1, 2)*SIZE(bvti1, 3)&
   &             )
   CALL POPREAL8ARRAY(bvti2, SIZE(bvti2, 1)*SIZE(bvti2, 2)*SIZE(bvti2, 3)&
   &             )
   CALL POPREAL8ARRAY(bmtj1, SIZE(bmtj1, 1)*SIZE(bmtj1, 2)*SIZE(bmtj1, 3)&
   &              *SIZE(bmtj1, 4))
   CALL POPREAL8ARRAY(bmtj2, SIZE(bmtj2, 1)*SIZE(bmtj2, 2)*SIZE(bmtj2, 3)&
   &              *SIZE(bmtj2, 4))
   CALL APPLYALLBC_BLOCK_B(.true.)
   CALL COMPUTEEDDYVISCOSITY_B()
   CALL POPREAL8ARRAY(p, SIZE(p, 1)*SIZE(p, 2)*SIZE(p, 3))
   CALL COMPUTELAMVISCOSITY_B()
   DO k=kb,0,-1
   DO j=jb,0,-1
   DO i=ib,0,-1
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   pinfcorrb = pinfcorrb + 1.e-4_realType*pb(i, j, k)
   pb(i, j, k) = 0.0_8
   END IF
   tempb = gm1*pb(i, j, k)
   wb(i, j, k, irhoe) = wb(i, j, k, irhoe) + tempb
   wb(i, j, k, irho) = wb(i, j, k, irho) - half*v2*tempb
   v2b = -(half*w(i, j, k, irho)*tempb)
   pb(i, j, k) = 0.0_8
   CALL POPREAL8(v2)
   wb(i, j, k, ivx) = wb(i, j, k, ivx) + 2*w(i, j, k, ivx)*v2b
   wb(i, j, k, ivy) = wb(i, j, k, ivy) + 2*w(i, j, k, ivy)*v2b
   wb(i, j, k, ivz) = wb(i, j, k, ivz) + 2*w(i, j, k, ivz)*v2b
   END DO
   END DO
   END DO
   CALL POPCONTROL1B(branch)
   IF (branch .EQ. 0) THEN
   CALL POPREAL8ARRAY(si, SIZE(si, 1)*SIZE(si, 2)*SIZE(si, 3)*SIZE(si, &
   &                4))
   CALL POPREAL8ARRAY(sj, SIZE(sj, 1)*SIZE(sj, 2)*SIZE(sj, 3)*SIZE(sj, &
   &                4))
   CALL POPREAL8ARRAY(sk, SIZE(sk, 1)*SIZE(sk, 2)*SIZE(sk, 3)*SIZE(sk, &
   &                4))
   CALL METRIC_BLOCK_B()
   CALL XHALO_BLOCK_B()
   END IF
   CALL SETFLOWINFINITYSTATE_B()
   CALL POPREAL8(gammainf)
   CALL POPREAL8(tref)
   CALL POPREAL8(pref)
   CALL POPREAL8(rhoref)
   CALL REFERENCESTATE_B()
   CALL ADJUSTINFLOWANGLE_B(alpha, alphab, beta, betab, liftindex)
   funcvaluesb = 0.0_8
   momentb = 0.0_8
   forceb = 0.0_8
   cavitationb = 0.0_8
   sepsensorb = 0.0_8
   END SUBROUTINE BLOCK_RES_B
