!------------------------------------------------------------------------------
! Module: argonne_v14_potential
!
! Purpose: Provides the Argonne V14 (AV14) nucleon-nucleon potential
!          based on Wiringa, Smith, and Ainsworth, Phys. Rev. C 29, 1207 (1984)
!
! Units: MeV for energy, fermi (fm) for distance
!------------------------------------------------------------------------------
module argonne_v14_potential
  implicit none
  private
  
  ! Public procedures
  public :: pot_v14
  
  ! Constants
  real(kind=8), parameter :: hbar_c = 197.33d0      ! MeV·fm
  real(kind=8), parameter :: nucleon_mass = 938.9d0 ! MeV/c²
  real(kind=8), parameter :: pion_mass = 138.03d0   ! MeV/c²

contains

  !----------------------------------------------------------------------------
  ! Main interface subroutine to calculate the Argonne V14 potential
  ! for a given radius
  !
  ! Input:
  !   radius - The radial distance in fermi (fm)
  !
  ! Output:
  !   potential - The AV14 potential matrix with dimensions:
  !               potential(L, L', S, J)
  !               L, L' = 0-5 (orbital angular momentum)
  !               S = 0,1 (spin)
  !               J = 0-4 (total angular momentum)
  !----------------------------------------------------------------------------
  subroutine pot_v14(radius, potential)
    implicit none
    real(kind=8), intent(in) :: radius
    real(kind=8), intent(out) :: potential(0:5, 0:5, 0:1, 0:4)
    
    ! Local variables
    integer, parameter :: nx = 1
    real(kind=8) :: v_uncoupled(nx, 10)
    real(kind=8) :: vw(nx, 10), vwp(nx, 10), vwpp(nx, 10), vws(nx, 10)
    real(kind=8) :: pot11(nx, 4), pot12(nx, 4), pot22(nx, 4)
    real(kind=8) :: vw11(nx, 4), vwp11(nx, 4), vwpp11(nx, 4), vws11(nx, 4)
    real(kind=8) :: hm
    
    ! Call the nuclear potential calculation routine
    call nnpot(radius, nx, hm, nx, &
               v_uncoupled, vw, vwp, vwpp, vws, &
               pot11, pot12, pot22, &
               vw11, vwp11, vwpp11, vws11)
    
    ! Set the output potential values
    ! Central uncoupled states
    potential(0, 0, 0, 0) = v_uncoupled(1, 1)
    potential(0, 0, 1, 1) = pot11(1, 1)
    potential(0, 2, 1, 1) = pot12(1, 1)
    potential(2, 0, 1, 1) = potential(0, 2, 1, 1)
    potential(2, 2, 1, 1) = pot22(1, 1)
    
    ! Complete the remaining elements
    potential(1, 1, 1, 0) = v_uncoupled(1, 2)
    potential(1, 1, 0, 1) = v_uncoupled(1, 3)
    potential(1, 1, 1, 1) = v_uncoupled(1, 4)
    potential(2, 2, 1, 2) = v_uncoupled(1, 5)
    potential(2, 2, 0, 2) = v_uncoupled(1, 6)
    potential(3, 3, 0, 3) = v_uncoupled(1, 7)
    potential(3, 3, 1, 3) = v_uncoupled(1, 8)
    potential(4, 4, 1, 4) = v_uncoupled(1, 9)
    potential(4, 4, 0, 4) = v_uncoupled(1, 10)
    potential(1, 1, 1, 2) = pot11(1, 2)
    potential(3, 3, 1, 2) = pot22(1, 2)
    potential(1, 3, 1, 2) = pot12(1, 2)
    potential(3, 1, 1, 2) = pot12(1, 2)
    potential(2, 2, 1, 3) = pot11(1, 3)
    potential(4, 4, 1, 3) = pot22(1, 3)
    potential(2, 4, 1, 3) = pot12(1, 3)
    potential(4, 2, 1, 3) = pot12(1, 3)
    potential(3, 3, 1, 4) = pot11(1, 4)
    potential(5, 5, 1, 4) = pot22(1, 4)
    potential(3, 5, 1, 4) = pot12(1, 4)
    potential(5, 3, 1, 4) = pot12(1, 4)
  end subroutine pot_v14

  !----------------------------------------------------------------------------
  ! Realistic AV14 nucleon-nucleon potentials in R-space
  !
  ! POT array contains: 1S0 3P0 1P1 3P1 3D2 1D2 1F3 3F3 3G4 1G4
  ! POT11 contains: 3S1 3P2 3D3 3F4
  ! POT22 contains: 3D1 3F2 3G3 3H4
  !----------------------------------------------------------------------------
  subroutine nnpot(radius, jmx, hm, nx, &
                   pot, vw, vwp, vwpp, vws, &
                   pot11, pot12, pot22, &
                   vw11, vwp11, vwpp11, vws11)
    implicit none
    real(kind=8), intent(in) :: radius
    integer, intent(in) :: jmx, nx
    real(kind=8), intent(out) :: hm
    real(kind=8), intent(out) :: pot(nx, 10), vw(nx, 10), vwp(nx, 10), vwpp(nx, 10), vws(nx, 10)
    real(kind=8), intent(out) :: pot11(nx, 4), pot12(nx, 4), pot22(nx, 4)
    real(kind=8), intent(out) :: vw11(nx, 4), vwp11(nx, 4), vwpp11(nx, 4), vws11(nx, 4)
    
    ! Local variables
    integer :: ill(14), iss(14), ijj(14), iii(14), koupl(14)
    
    ! Define two-body states (L,S,J,I)
    ! KOUPL = 0 for uncoupled state, = 1 for coupled state
    ill = [0, 0, 1, 1, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3]
    iss = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    ijj = [0, 1, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    iii = [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    koupl = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    
    ! Call the Argonne potential generator
    call argonne(radius, jmx, hm, ill, iss, ijj, iii, koupl, nx, &
                 pot, vw, vwp, vwpp, pot11, pot12, pot22, vw11, vwp11, vwpp11)
    
    ! Note: VW, VWP, VWPP, VWS, VW11, VWP11, VWPP11 and VWS11 are used
    ! only for velocity-dependent potential (such as Bonn and Paris)
  end subroutine nnpot

  !----------------------------------------------------------------------------
  ! Potential generator for Argonne potential (the V14 model)
  ! Reference: R.B. Wiringa et al., Phys. Rev. C 29 (1984) 1207
  !----------------------------------------------------------------------------
  subroutine argonne(radius, jmx, hm, ill, iss, ijj, iii, koupl, nx, &
                    pot, vw, vwp, vwpp, pot11, pot12, pot22, vw11, vwp11, vwpp11)
    implicit none
    real(kind=8), intent(in) :: radius
    integer, intent(in) :: jmx, nx
    real(kind=8), intent(out) :: hm
    integer, intent(in) :: ill(14), iss(14), ijj(14), iii(14), koupl(14)
    real(kind=8), intent(out) :: pot(nx, 10), vw(nx, 10), vwp(nx, 10), vwpp(nx, 10)
    real(kind=8), intent(out) :: pot11(nx, 4), pot12(nx, 4), pot22(nx, 4)
    real(kind=8), intent(out) :: vw11(nx, 4), vwp11(nx, 4), vwpp11(nx, 4)
    
    ! Local variables
    integer, parameter :: nz = 5000
    real(kind=8) :: pv(14, nz)
    real(kind=8) :: ca(14, 10), ca11(14, 4), ca12(5:6, 4), ca22(14, 4)
    integer :: k, ku, kc, jx, ll, is, jj, ii, icpl, ip
    real(kind=8) :: sum, sum11, sum22, sum12
    
    ! Initialize counters for uncoupled and coupled states
    ku = 0
    kc = 0
    
    ! Calculate coefficients CA
    do k = 1, 14
      ll = ill(k)
      is = iss(k)
      jj = ijj(k)
      ii = iii(k)
      icpl = koupl(k)
      
      if (icpl == 0) then
        ! Uncoupled states
        ku = ku + 1
        ca(1, ku) = 1.0d0
        ca(2, ku) = real(4*ii-3, kind=8)
        ca(3, ku) = real(4*is-3, kind=8)
        ca(4, ku) = ca(2, ku) * ca(3, ku)
        ca(5, ku) = 0.0d0
        if (ll == jj .and. is == 1) ca(5, ku) = 2.0d0
        if (ku == 2) ca(5, ku) = -4.0d0
        ca(6, ku) = ca(5, ku) * ca(2, ku)
        ca(7, ku) = 0.5d0 * real(jj*(jj+1) - ll*(ll+1) - is*(is+1), kind=8)
        ca(8, ku) = ca(7, ku) * ca(2, ku)
        ca(9, ku) = real(ll*(ll+1), kind=8)
        ca(10, ku) = ca(9, ku) * ca(2, ku)
        ca(11, ku) = ca(9, ku) * ca(3, ku)
        ca(12, ku) = ca(10, ku) * ca(3, ku)
        ca(13, ku) = ca(7, ku) * ca(7, ku)
        ca(14, ku) = ca(13, ku) * ca(2, ku)
      else
        ! Coupled states
        kc = kc + 1
        ca11(1, kc) = 1.0d0
        ca22(1, kc) = 1.0d0
        ca11(2, kc) = real(4*ii-3, kind=8)
        ca22(2, kc) = ca11(2, kc)
        ca11(3, kc) = real(4*is-3, kind=8)
        ca22(3, kc) = ca11(3, kc)
        ca11(4, kc) = ca11(2, kc) * ca11(3, kc)
        ca22(4, kc) = ca11(4, kc)
        ca11(5, kc) = -2.0d0 * real(jj-1, kind=8) / real(2*jj+1, kind=8)
        ca12(5, kc) = 6.0d0 * sqrt(real(jj*(jj+1), kind=8)) / real(2*jj+1, kind=8)
        ca22(5, kc) = -2.0d0 * real(jj+2, kind=8) / real(2*jj+1, kind=8)
        ca11(6, kc) = ca11(5, kc) * ca11(2, kc)
        ca12(6, kc) = ca12(5, kc) * ca11(2, kc)
        ca22(6, kc) = ca22(5, kc) * ca11(2, kc)
        ca11(7, kc) = 0.5d0 * real(jj*(jj+1) - ll*(ll+1) - is*(is+1), kind=8)
        ca22(7, kc) = 0.5d0 * real(jj*(jj+1) - (ll+2)*(ll+3) - is*(is+1), kind=8)
        ca11(8, kc) = ca11(7, kc) * ca11(2, kc)
        ca22(8, kc) = ca22(7, kc) * ca22(2, kc)
        ca11(9, kc) = real(ll*(ll+1), kind=8)
        ca22(9, kc) = real((ll+2)*(ll+3), kind=8)
        ca11(10, kc) = ca11(9, kc) * ca11(2, kc)
        ca22(10, kc) = ca22(9, kc) * ca22(2, kc)
        ca11(11, kc) = ca11(9, kc) * ca11(3, kc)
        ca22(11, kc) = ca22(9, kc) * ca22(3, kc)
        ca11(12, kc) = ca11(10, kc) * ca11(3, kc)
        ca22(12, kc) = ca22(10, kc) * ca22(3, kc)
        ca11(13, kc) = ca11(7, kc) * ca11(7, kc)
        ca22(13, kc) = ca22(7, kc) * ca22(7, kc)
        ca11(14, kc) = ca11(13, kc) * ca11(2, kc)
        ca22(14, kc) = ca22(13, kc) * ca22(2, kc)
      end if
    end do
    
    ! Calculate potential values
    call av14(radius, pv, jmx)
    
    ! Calculate uncoupled state potentials
    do k = 1, 10
      do jx = 1, jmx
        sum = 0.0d0
        do ip = 1, 14
          sum = sum + ca(ip, k) * pv(ip, jx)
        end do
        pot(jx, k) = sum
        vw(jx, k) = 0.0d0
        vwp(jx, k) = 0.0d0
        vwpp(jx, k) = 0.0d0
      end do
    end do
    
    ! Calculate coupled state potentials
    do k = 1, 4
      do jx = 1, jmx
        sum11 = 0.0d0
        sum22 = 0.0d0
        do ip = 1, 14
          sum11 = sum11 + ca11(ip, k) * pv(ip, jx)
          sum22 = sum22 + ca22(ip, k) * pv(ip, jx)
        end do
        pot11(jx, k) = sum11
        pot22(jx, k) = sum22
        
        sum12 = 0.0d0
        do ip = 5, 6
          sum12 = sum12 + ca12(ip, k) * pv(ip, jx)
        end do
        pot12(jx, k) = sum12
        
        vw11(jx, k) = 0.0d0
        vwp11(jx, k) = 0.0d0
        vwpp11(jx, k) = 0.0d0
      end do
    end do
    
    ! Set the h-bar squared over mass constant (MeV·fm²)
    hm = nucleon_mass / hbar_c**2
  end subroutine argonne

  !----------------------------------------------------------------------------
  ! Calculate the AV14 potential components
  !----------------------------------------------------------------------------
  subroutine av14(radius, pv, jmx)
    implicit none
    real(kind=8), intent(in) :: radius
    integer, intent(in) :: jmx
    real(kind=8), intent(out) :: pv(14, *)
    
    ! Local variables
    integer :: jx
    real(kind=8) :: amu, x, rcut, ypi, tpi, tpi2, w
    
    ! Pion mass in inverse fermi
    amu = pion_mass / hbar_c
    
    do jx = 1, jmx
      ! Calculate scaled radius
      x = amu * radius
      
      ! Calculate cutoff function
      rcut = 1.0d0
      if (radius < 6.0d0) rcut = 1.0d0 - exp(-2.0d0 * radius**2)
      
      ! Calculate Yukawa functions
      ypi = exp(-x) * rcut / x
      tpi = (1.0d0 + 3.0d0/x + 3.0d0/x**2) * ypi * rcut
      tpi2 = tpi * tpi
      
      ! Calculate short-range function
      w = 0.0d0
      if (radius < 10.0d0) w = 1.0d0 / (1.0d0 + exp((radius - 0.5d0) / 0.2d0))
      
      ! Set potential components
      pv(1, jx) = -4.801125d0 * tpi2 + 2061.5625d0 * w
      pv(2, jx) = 0.798925d0 * tpi2 - 477.3125d0 * w
      pv(3, jx) = 1.189325d0 * tpi2 - 502.3125d0 * w
      pv(4, jx) = 3.72681d0 * ypi + 0.182875d0 * tpi2 + 97.0625d0 * w
      pv(5, jx) = -0.1575d0 * tpi2 + 108.75d0 * w
      pv(6, jx) = 3.72681d0 * tpi - 0.7525d0 * tpi2 + 297.25d0 * w
      pv(7, jx) = 0.5625d0 * tpi2 - 719.75d0 * w
      pv(8, jx) = 0.0475d0 * tpi2 - 159.25d0 * w
      pv(9, jx) = 0.070625d0 * tpi2 + 8.625d0 * w
      pv(10, jx) = -0.148125d0 * tpi2 + 5.625d0 * w
      pv(11, jx) = -0.040625d0 * tpi2 + 17.375d0 * w
      pv(12, jx) = -0.001875d0 * tpi2 - 33.625d0 * w
      pv(13, jx) = -0.5425d0 * tpi2 + 391.0d0 * w
      pv(14, jx) = 0.0025d0 * tpi2 + 145.0d0 * w
    end do
  end subroutine av14

end module argonne_v14_potential