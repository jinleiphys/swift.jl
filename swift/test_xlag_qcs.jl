# test_xlag_qcs.jl — Step 2 (option A): put the Lagrange-Laguerre coordinate through the SAME
# basis-agnostic q-operator + smooth-ECS layer as the spline y, so x can carry an interior-real /
# exterior-rotated contour (needed to damp the breakup tail in x, Jin 2026-06-30).
#
# Two checks:
#   (1) lagrange_laguerre_regularized_derivs value matches lagrange_laguerre_regularized_basis,
#       and ∂/∂² match central finite differences (a few digits).
#   (2) Lagrange-Laguerre GALERKIN MT ¹S₀ 2-body scattering through the q-operator contour layer
#       reproduces δ≈63.5°, η→1 (smooth ECS) — same target as the spline Galerkin (test_yspline_galerkin.jl).
#       No BC dropping: the regularized Lagrange-Laguerre basis is regular at 0 and decays at ∞.

using LinearAlgebra, Printf, FastGaussQuadrature
const BB = "/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/swift/laguerre.jl");           using .Laguerre
include(BB*"/swift/ecs.jl");                using .ECS
include(BB*"/general_modules/mesh.jl");     using .mesh

const ħ = 197.3269718; const amu = 931.49432; const mN = 1.0079713395678829
const μ = mN/2.0; const ħ2_2μ = ħ^2/(2.0*μ*amu)
mt_1S0(z) = 1438.72*exp(-3.11*z)/z - 513.968*exp(-1.55*z)/z
F0(z) = sin(z)

# ---- build a 1D Laguerre x-mesh (reuse initialmesh, use the x part) ----
function lag_mesh(nx, xmax; alpha=1.0)
    g = initialmesh(4, nx, 4, Float64(xmax), Float64(xmax), Float64(alpha))
    return g
end

# ---- check (1): derivatives vs finite differences ----
function check_derivs()
    g = lag_mesh(20, 40.0)
    xi, ϕx, a, hs = g.xi, g.ϕx, g.α, g.hsx
    r = 7.3                                  # generic point, not a node
    f, f1, f2 = lagrange_laguerre_regularized_derivs(r, xi, ϕx, a, hs)
    fval = real.(lagrange_laguerre_regularized_basis(r, xi, ϕx, a, hs))
    h = 1e-4
    fp = real.(lagrange_laguerre_regularized_basis(r+h, xi, ϕx, a, hs))
    fm = real.(lagrange_laguerre_regularized_basis(r-h, xi, ϕx, a, hs))
    fd1 = (fp .- fm) ./ (2h); fd2 = (fp .- 2 .* fval .+ fm) ./ h^2
    @printf("  value   max|Δ| vs basis()      : %.2e\n", maximum(abs, f .- fval))
    @printf("  1st der max|Δ| vs central FD   : %.2e\n", maximum(abs, f1 .- fd1))
    @printf("  2nd der max|Δ| vs central FD   : %.2e\n", maximum(abs, f2 .- fd2))
end

# ---- check (2): Lagrange-Laguerre Galerkin q-operator MT ¹S₀ ----
function solve_lag_2body(cs::CSContour, E, nx, xmax; ngfac=5, l=0)
    k = sqrt(E/ħ2_2μ)
    g = lag_mesh(nx, xmax); xi, ϕx, a, hs = g.xi, g.ϕx, g.α, g.hsx
    ng = ngfac*nx
    uq, wq = gausslegendre(ng)
    rq = (uq .+ 1.0) .* (xmax/2.0); wr = wq .* (xmax/2.0)
    S = zeros(ComplexF64, nx, nx); K = zeros(ComplexF64, nx, nx)
    V = zeros(ComplexF64, nx, nx); b = zeros(ComplexF64, nx)
    for qd in 1:ng
        r = rq[qd]; w = wr[qd]
        x = contour_x(cs, r); q = contour_q(cs, r); qp = contour_qp(cs, r)
        f, f1, f2 = lagrange_laguerre_regularized_derivs(r, xi, ϕx, a, hs)
        Vx = mt_1S0(x); cent = ħ2_2μ*l*(l+1)/x^2; wqq = w*q
        for i in 1:nx
            b[i] += wqq*f[i]*Vx*F0(k*x)
            for j in 1:nx
                S[i,j] += wqq*f[i]*f[j]
                kin = ħ2_2μ*((1.0/q^2)*f2[j] - (qp/q^3)*f1[j])
                K[i,j] += wqq*f[i]*(kin - cent*f[j])
                V[i,j] += wqq*f[i]*Vx*f[j]
            end
        end
    end
    A = E .* S .+ K .- V
    c = A \ b
    resid = norm(A*c - b)/norm(b)
    # amplitude: bra = regular F0, ∫·q dr
    M = 0.0im
    for qd in 1:ng
        r = rq[qd]; w = wr[qd]
        x = contour_x(cs, r); q = contour_q(cs, r)
        f, _, _ = lagrange_laguerre_regularized_derivs(r, xi, ϕx, a, hs)
        u = F0(k*x) + sum(c[i]*f[i] for i in 1:nx)
        M += w*q*F0(k*x)*mt_1S0(x)*u
    end
    f_amp = -(1.0/E)*M; Sm = 1.0 + 2im*k*f_amp
    δ = rad2deg(0.5*angle(Sm)); δ = δ<0 ? δ+180 : δ
    return δ, abs(Sm), resid
end

println("="^70)
println(" (1) Lagrange-Laguerre real-r derivatives vs finite differences")
println("="^70)
check_derivs()

println("\n"*"="^70)
println(" (2) Lagrange-Laguerre Galerkin + SMOOTH-ECS, MT ¹S₀ (target δ≈63.5°, η→1)")
println(" R0=6, w=2; θ-independence + η→1 is the validation")
println("="^70)
@printf(" %4s | %10s %10s %10s\n", "θ°", "δ(deg)", "η", "resid")
for θd in (8.0, 10.0, 12.0, 14.0, 16.0)
    cs = CSContour(:smooth; θ_deg=θd, R0=6.0, w=2.0)
    δ, η, res = solve_lag_2body(cs, 1.0, 60, 100.0)
    @printf(" %4.0f | %10.3f %10.4f %10.2e\n", θd, δ, η, res)
end
