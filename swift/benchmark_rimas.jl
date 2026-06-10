#!/usr/bin/env julia
# Cross-validation scan vs Lazauskas (rimas) benchmark tables, AV18 no 3NF, T=1/2.
# Scans jx_max (= j2bmax) = 1..6 for 3H (MT=-1/2) and 3He (MT=+1/2, point Coulomb).
# Convention note: swift.jl uses hbar^2/m_N = 41.468 MeV fm^2 vs rimas 41.471
# => expect a systematic ~5-6 keV under-binding, NOT a bug.

using LinearAlgebra
using Printf
BLAS.set_num_threads(Sys.CPU_THREADS)
println("BLAS threads: $(BLAS.get_num_threads())")

include("../general_modules/channels.jl")
include("../general_modules/mesh.jl")
using .channels
using .mesh
include("twobody.jl")
using .twobodybound
include("MalflietTjon.jl")
using .MalflietTjon

# Same setup as the 2026-05-21 jx_max<=3 cross-check
fermion = true; Jtot = 0.5; T = 0.5; Parity = 1
lmax = 8; lmin = 0; λmax = 20; λmin = 0
s1 = 0.5; s2 = 0.5; s3 = 0.5; t1 = 0.5; t2 = 0.5; t3 = 0.5
nθ = 12; nx = 20; ny = 20; xmax = 16.0; ymax = 16.0; alpha = 1.0
potname = "AV18"

# rimas tables (personal communication, 2026-05-21): jx => (N_AMP, E)
rimas = Dict(
    "3H"  => Dict(1 => (10, -7.196), 2 => (18, -7.502), 3 => (26, -7.594),
                  4 => (34, -7.606), 5 => (42, -7.614), 6 => (50, -7.615)),
    "3He" => Dict(1 => (10, -6.499), 2 => (18, -6.809), 3 => (26, -6.897),
                  4 => (34, -6.907), 5 => (42, -6.916), 6 => (50, -6.917)),
)

grid = initialmesh(nθ, nx, ny, xmax, ymax, alpha)
e2b, _ = bound2b(grid, potname)   # np deuteron, shared by both systems
@printf("Deuteron E2b = %.6f MeV\n", e2b[1])

# Pre-compile with tiny grid
α_test = α3b(fermion, Jtot, T, Parity, 2, 0, 2, 0, s1, s2, s3, t1, t2, t3, -0.5, 1.0)
grid_test = initialmesh(4, 5, 5, 10.0, 10.0, 1.0)
e2b_test, _ = bound2b(grid_test, potname)
_ = malfiet_tjon_solve_optimized(α_test, grid_test, potname, e2b_test,
    E0=-7.5, E1=-6.5, tolerance=1e-3, max_iterations=1, verbose=false, include_uix=false)
println("Pre-compilation done.\n")

rows = []
for (sys, MT) in [("3H", -0.5), ("3He", +0.5)]
    println("="^72)
    println("  $sys  (MT = $MT)")
    println("="^72)
    for jx in 1:6
        N_ref, E_ref = rimas[sys][jx]
        α = α3b(fermion, Jtot, T, Parity, lmax, lmin, λmax, λmin,
                s1, s2, s3, t1, t2, t3, MT, Float64(jx))
        t = @elapsed begin
            result, _, _ = malfiet_tjon_solve_optimized(α, grid, potname, e2b,
                E0=E_ref - 0.3, E1=E_ref + 0.3,
                tolerance=1e-6, max_iterations=30, verbose=false, include_uix=false)
        end
        Δ = (result.energy - E_ref) * 1000  # keV
        push!(rows, (sys, jx, α.nchmax, N_ref, result.energy, E_ref, Δ, result.converged, t))
        @printf("%-4s jx=%d  N=%3d (rimas %3d)  E = %+.4f MeV  (rimas %+.4f)  Δ = %+6.1f keV  conv=%s  [%.0f s]\n",
                sys, jx, α.nchmax, N_ref, result.energy, E_ref, Δ, result.converged, t)
        flush(stdout)
    end
end

println("\n" * "="^72)
println("  SUMMARY  (Δ = E_swift - E_rimas; expect ~ +5 keV from m_N convention)")
println("="^72)
@printf("%-5s %-3s %8s %8s %12s %12s %9s %6s\n",
        "sys", "jx", "N_swift", "N_rimas", "E_swift", "E_rimas", "Δ (keV)", "conv")
for r in rows
    @printf("%-5s %-3d %8d %8d %12.4f %12.4f %+9.1f %6s\n",
            r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
end
