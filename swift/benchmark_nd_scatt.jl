# benchmark_nd_scatt.jl
# n-d elastic scattering cross-validation vs Lazauskas-Carbonell PRC 84, 034002 (2011), Table III.
# Potential: MT I-III (central, S-wave only).  T=1/2 approx, equal masses, hbar^2/m_N = 41.47 MeV fm^2.
#
# Benchmark Re(delta), eta (inelasticity = |S|) for n-d L=0:
#   doublet (S=1/2, Jtot=1/2):  E_lab=14.1 -> 105.49 deg, eta=0.4649 ;  E_lab=42 -> 41.35 deg, eta=0.5022
#   quartet (S=3/2, Jtot=3/2):  E_lab=14.1 ->  68.95 deg, eta=0.9782 ;  E_lab=42 -> 37.71 deg, eta=0.9033
#
# Rimas guidance: do NOT benchmark CS at 1 MeV (q too small, CS wf decays as exp(-q sin(theta) y),
# needs huge dense grids). Stay in his converged theta window: [4,12.5] deg @14.1, [3,7.5] deg @42.
# CS angle hard limit (Eq.18): theta_max = 14.2 deg @14.1 MeV, 8.9 deg @42 MeV.

using LinearAlgebra
using Printf

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
include("matrices_optimized.jl")
include("scattering.jl")
include("twobody.jl")

using .channels
using .mesh
using .matrices_optimized
using .Scattering
using .twobodybound

const ħ   = 197.3269718          # MeV*fm
const m   = 1.0079713395678829   # amu
const amu = 931.49432            # MeV
# n-d reduced mass mu = m_n m_d/(m_n+m_d) = 2m/3 ; hbar^2/m_N = 41.47 MeV fm^2 check:
# ħ^2/(m*amu) = 197.327^2/(1.00797*931.494) = 41.47  OK

"""
Run one n-d elastic scattering point and return (E_d, results-dict).
channel_spin_target = (λ, 𝕊) of the elastic entrance channel.
"""
function run_point(; Jtot, E_lab, θ_deg, potname="MT",
                     lmax=2, λmax=2, j2bmax=1.0,
                     nθ=12, nx=30, ny=70, xmax=30.0, ymax=60.0, alpha=1.0)
    fermion = true
    T = 0.5; Parity = 1; MT = -0.5          # n+d
    lmin = 0; λmin = 0
    s1=0.5; s2=0.5; s3=0.5; t1=0.5; t2=0.5; t3=0.5
    z1z2 = 0.0                               # neutral n+d, no Coulomb

    # n+d kinematics: E_cm = E_lab * m_d/(m_n+m_d) = (2/3) E_lab  (equal masses)
    E = (2.0/3.0) * E_lab

    α    = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
               s1,s2,s3, t1,t2,t3, MT, j2bmax)
    grid = initialmesh(nθ, nx, ny, xmax, ymax, alpha)

    # Scaled V for the amplitude matrix element (must match the complex-scaled solve).
    V              = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
    Rxy, Rxy_31    = Rxy_matrix_optimized(α, grid)

    bE, bψ = bound2b(grid, potname, θ_deg=θ_deg)
    isempty(bE) && error("no deuteron bound state")
    φ_d_matrix = ComplexF64.(bψ[1])
    E_d        = real(bE[1])

    θ_rad = θ_deg * π / 180.0
    # E = E_cm = n-d relative kinetic energy (sets the incoming momentum q and the
    # amplitude k). The Faddeev OPERATOR uses the total 3-body energy E_total = E_cm + E_d
    # (E_d < 0), i.e. the deuteron binding must enter the energy balance.
    E_total = E + E_d
    ψ_in  = compute_initial_state_vector(grid, α, φ_d_matrix, E, z1z2, θ=θ_rad)
    ψ_sc, A, b = solve_scattering_equation(E_total, α, grid, potname, ψ_in,
                                           θ_deg=θ_deg)
    res_norm = norm(A*ψ_sc - b)
    rel_res  = res_norm / norm(b)

    f, dch, labels = compute_scattering_amplitude(ψ_in, V, Rxy_31, ψ_sc, E, grid, α,
                                                  φ_d_matrix, z1z2, θ=θ_rad, σ_l=0.0)

    μ = (2.0*m)/3.0
    k = sqrt(2.0 * μ * amu * E) / ħ

    # collision matrix -> channel-spin basis ; pull elastic (λ,𝕊) diagonal element
    U          = compute_collision_matrix(f, k)
    U_cs, lab  = Scattering.recouple_to_channel_spin(U, α, dch)
    return (E=E, E_d=E_d, k=k, res_norm=res_norm, rel_res=rel_res, nch=α.nchmax,
            U_cs=U_cs, lab=lab)
end

"From the channel-spin U block, get (delta_deg, eta) for the elastic entrance channel label."
function elastic_delta_eta(U_cs, lab, Jtot, parity, target_label)
    key = (Jtot, parity)
    haskey(U_cs, key) || error("no (J=$Jtot, π=$parity) block; have $(keys(U_cs))")
    U = U_cs[key]; L = lab[key]
    idx = findfirst(==(target_label), L)
    idx === nothing && error("label '$target_label' not in $L")
    s   = U[idx, idx]                  # elastic S-matrix element S = eta * exp(2i delta)
    δ   = 0.5 * angle(s)
    η   = abs(s)
    return rad2deg(δ), η, L
end

# --------------------------------------------------------------------------
println("="^78)
println("  n-d elastic scattering  vs  Lazauskas-Carbonell PRC 84, 034002 (2011) Tab.III")
println("="^78)

# Channel-convergence scan, DOUBLET @ E_lab=14.1 MeV, theta=10 deg.
# Reduced mesh (dense A is nch·nx·ny squared) so we can grow the channel space and
# watch whether δ → 105.49 and η → 0.4649 (≤1). If η stays > 1 as channels grow,
# the unitarity violation is NOT just truncation.
Jtot = 0.5; E_lab = 14.1; θ = 10.0
println("\nDOUBLET  Jtot=1/2  E_lab=$(E_lab) MeV (E_cm=$(round((2/3)*E_lab,digits=3)))  θ=$(θ)°")
println("benchmark: Re(δ)=105.49°, η=0.4649")
@printf("\n%-6s %-6s %-6s %5s %10s %10s %10s\n", "lmax","λmax","j2b","nch","Re(δ)","η","rel_res")
for (lmax, λmax, j2b) in [(2,2,1.0), (3,4,2.0), (4,6,2.0)]
    r = run_point(Jtot=Jtot, E_lab=E_lab, θ_deg=θ, lmax=lmax, λmax=λmax, j2bmax=j2b,
                  nx=12, ny=30, xmax=24.0, ymax=50.0)
    δ, η, _ = elastic_delta_eta(r.U_cs, r.lab, Jtot, 1, "λ=0, 𝕊=0.5")
    @printf("%-6d %-6d %-6.0f %5d %10.3f %10.4f %10.1e\n", lmax, λmax, j2b, r.nch, δ, η, r.rel_res)
    flush(stdout)
end
println("="^78)
