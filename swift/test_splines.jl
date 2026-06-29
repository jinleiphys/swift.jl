# Self-test for the Hermite-spline basis (swift/splines.jl).
# Mirrors the NNpot/test.jl smoke-test style: each block checks a property that must
# hold EXACTLY (to machine precision), so a failure points straight at the broken piece.
#
# Run:  cd swift && julia test_splines.jl

include("splines.jl")
using .Splines
using Printf

println("="^70)
println(" Hermite-spline basis self-test (port of Lazauskas SPL/SPL_CMP/COLLOC)")
println("="^70)

# A graded grid: dense inner domain, geometric outer domain, on [0, 60].
domains = [(0.0, 20.0, 30, 1.0), (20.0, 60.0, 25, 1.06)]

# Global DOF vector for a known function f: node j stores (f, f', f'') at knots[j].
function dof_vector(mesh, f, fp, fpp)
    c = zeros(mesh.ndof)
    for (j, x) in enumerate(mesh.knots)
        b = (j - 1) * mesh.ncol
        c[b + 1] = f(x)
        c[b + 2] = fp(x)
        mesh.ncol == 3 && (c[b + 3] = fpp(x))
    end
    return c
end

interp(mesh, c, r; θ = 0.0) = begin
    idx, S, _, _ = spline_functions(mesh, r; θ = θ)
    sum(c[idx] .* S)
end
interp_d1(mesh, c, r) = begin
    idx, _, S1, _ = spline_functions(mesh, r)
    sum(c[idx] .* S1)
end
interp_d2(mesh, c, r) = begin
    idx, _, _, S2 = spline_functions(mesh, r)
    sum(c[idx] .* S2)
end

test_points = [0.37, 3.1, 12.9, 20.0, 21.4, 41.7, 59.3]

for ncol in (2, 3)
    order = ncol == 2 ? "cubic" : "quintic"
    deg   = 2ncol - 1   # exactly-reproduced polynomial degree
    println("\n--- ncol = $ncol ($order Hermite, reproduces degree <= $deg) ---")
    mesh = init_spline_mesh(domains; ncol = ncol)
    @printf("  intervals N = %d, raw DOF = %d, collocation points = %d\n",
            mesh.nint, mesh.ndof, length(mesh.xc))

    # 1) Collocation count must be ncol per interval.
    @assert length(mesh.xc) == ncol * mesh.nint
    println("  [ok] collocation count = ncol*N")

    # 2) Partition of unity: interpolating the constant 1 returns 1 everywhere.
    c1 = dof_vector(mesh, r -> 1.0, r -> 0.0, r -> 0.0)
    pou = maximum(abs(interp(mesh, c1, r) - 1.0) for r in test_points)
    @printf("  [%s] partition of unity, max|Σφ - 1| = %.2e\n", pou < 1e-13 ? "ok" : "XX", pou)
    @assert pou < 1e-13

    # 3) Polynomial reproduction to machine precision (value, 1st, 2nd derivative).
    co = [0.7, -0.3, 0.11, -0.02, 0.013, -0.0017][1:deg+1]
    f(r)   = sum(co[k] * r^(k-1)        for k in 1:deg+1)
    fp(r)  = sum(co[k] * (k-1) * r^(k-2) for k in 2:deg+1)
    fpp(r) = sum(co[k] * (k-1) * (k-2) * r^(k-3) for k in 3:deg+1)
    c = dof_vector(mesh, f, fp, fpp)
    # Relative errors (the degree-deg polynomial reaches ~rmax^deg ~ 1e8, so the honest
    # machine-precision metric is err / scale, not absolute err).
    scale0 = maximum(abs(f(r))   for r in test_points)
    scale1 = maximum(abs(fp(r))  for r in test_points)
    scale2 = maximum(abs(fpp(r)) for r in test_points)
    e0 = maximum(abs(interp(mesh, c, r)    - f(r))   for r in test_points) / scale0
    e1 = maximum(abs(interp_d1(mesh, c, r) - fp(r))  for r in test_points) / scale1
    e2 = maximum(abs(interp_d2(mesh, c, r) - fpp(r)) for r in test_points) / scale2
    @printf("  [%s] value   reproduction, max rel err = %.2e\n", e0 < 1e-12 ? "ok" : "XX", e0)
    @printf("  [%s] 1st-deriv reproduction, max rel err = %.2e\n", e1 < 1e-12 ? "ok" : "XX", e1)
    @printf("  [%s] 2nd-deriv reproduction, max rel err = %.2e\n", e2 < 1e-12 ? "ok" : "XX", e2)
    @assert e0 < 1e-12 && e1 < 1e-12 && e2 < 1e-12

    # 4) Complex scaling: for a polynomial, the CS interpolant must equal the analytic
    #    continuation f(r*e^{iθ}) (a polynomial is its own continuation).
    θ = 7.0 * π / 180
    ecs = maximum(abs(interp(mesh, c, r; θ = θ) - f(r * cis(θ))) for r in test_points) / scale0
    @printf("  [%s] CS continuation θ=7°, max rel err = %.2e\n", ecs < 1e-12 ? "ok" : "XX", ecs)
    @assert ecs < 1e-12

    # 5) Locality: at any interior point exactly 2*ncol DOF are active.
    idx, _, _, _ = spline_functions(mesh, 12.9)
    @assert length(idx) == 2ncol
    println("  [ok] locality: exactly 2*ncol active DOF per point")
end

println("\n" * "="^70)
println(" ALL SPLINE SELF-TESTS PASSED")
println("="^70)
