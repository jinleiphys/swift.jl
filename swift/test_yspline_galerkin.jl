# test_yspline_galerkin.jl — Step 2 foundation (Jin 2026-06-30).
#
# The 3-body uses a GALERKIN discretization (square T/V/B matrices, kron(x,y), then
# eigen(H,B) or A·ψ=b). The validated spline 2-body (test_2body_spline.jl solve_spline_2body_qcs)
# is COLLOCATION. Before wiring the spline+smooth-ECS y-coordinate into the 3-body kron, this
# file rebuilds the SAME MT ¹S₀ 2-body scattering in the GALERKIN spline form via the
# basis-agnostic q-operator contour layer (ecs.jl), and checks it reproduces the known answer
# (δ≈63.5°, η→1 with smooth ECS). This validates the spline Galerkin overlap S, the q-operator
# kinetic K (with the contour metric q,q'), the centrifugal term, and the BC handling — exactly
# the pieces that become Sy / Ky in the 3-body T_y block.
#
# Weak form (bilinear c-product ∫·q dr, NO conjugation under CS):
#   A[i,j] = E·S[i,j] + (ħ²/2μ)∫φ_i[(1/q²)φ_j'' − (q'/q³)φ_j'] q dr
#            − (ħ²/2μ) l(l+1) ∫φ_i φ_j /x(r)² q dr  − ∫φ_i V(x(r)) φ_j q dr
#   b[i]   = ∫ φ_i V(x(r)) F0(k·x(r)) q dr ;  solve A c = b (BC DOF dropped)
#   amplitude bra = regular F0 (NOT φ_i): f = −(1/E)∫ F0(k·x) V(x) u_tot q dr, u_tot=u_in+Σcφ.

using LinearAlgebra, Printf, FastGaussQuadrature
include("splines.jl"); using .Splines
include("ecs.jl");     using .ECS

const ħ   = 197.3269718
const amu = 931.49432
const mN  = 1.0079713395678829
const μ   = mN / 2.0
const ħ2_2μ = ħ^2 / (2.0 * μ * amu)        # = 41.471 MeV·fm²

mt_1S0(z) = 1438.72 * exp(-3.11 * z) / z - 513.968 * exp(-1.55 * z) / z
F0(z) = sin(z)                             # Riccati-Bessel regular, η=0, l=0

"""
    build_spline_galerkin_1d(mesh, cs, l; nq=8)
       -> (S, K, getrows) where K already includes kinetic + centrifugal in q-operator form.

Square (ndof×ndof) ComplexF64 Galerkin matrices on the contour `cs`, no BC applied yet.
`S[i,j]=∫φ_iφ_j q dr`, `K[i,j]=(ħ²/2μ)∫φ_i[(1/q²)φ_j''−(q'/q³)φ_j'] q dr + centrifugal`.
"""
function build_spline_galerkin_1d(mesh::SplineMesh, cs::CSContour, l::Int; nq::Int = 8)
    nd = mesh.ndof
    S = zeros(ComplexF64, nd, nd)
    K = zeros(ComplexF64, nd, nd)
    uq, wq = gausslegendre(nq)
    for ip in 1:mesh.nint
        a = mesh.knots[ip]; h = mesh.knots[ip + 1] - mesh.knots[ip]
        for g in 1:nq
            r = a + 0.5h * (uq[g] + 1.0); w = 0.5h * wq[g]
            x  = contour_x(cs, r); q = contour_q(cs, r); qp = contour_qp(cs, r)
            idx, φ, φ1, φ2 = spline_functions(mesh, r)        # θ=0: real-r value/∂/∂²
            cent = ħ2_2μ * l * (l + 1) / x^2
            wq_q = w * q                                       # ∫·q dr
            for (li, i) in enumerate(idx), (lj, j) in enumerate(idx)
                S[i, j] += wq_q * φ[li] * φ[lj]
                kin = ħ2_2μ * ((1.0 / q^2) * φ2[lj] - (qp / q^3) * φ1[lj])
                K[i, j] += wq_q * φ[li] * (kin - cent * φ[lj])
            end
        end
    end
    return S, K
end

"Galerkin MT ¹S₀ 2-body CS scattering through the q-operator contour layer."
function solve_galerkin_2body(cs::CSContour, E, domains; ncol = 3, nq = 8, l = 0)
    k = sqrt(E / ħ2_2μ)
    mesh = init_spline_mesh(domains; ncol = ncol)
    nd = mesh.ndof; nint = mesh.nint
    S, K = build_spline_galerkin_1d(mesh, cs, l; nq = nq)
    # V matrix and source via the same element quadrature
    V = zeros(ComplexF64, nd, nd); b = zeros(ComplexF64, nd)
    uq, wq = gausslegendre(nq)
    for ip in 1:nint
        a = mesh.knots[ip]; h = mesh.knots[ip + 1] - mesh.knots[ip]
        for g in 1:nq
            r = a + 0.5h * (uq[g] + 1.0); w = 0.5h * wq[g]
            x = contour_x(cs, r); q = contour_q(cs, r)
            Vx = mt_1S0(x); idx, φ, _, _ = spline_functions(mesh, r)
            wqq = w * q
            for (li, i) in enumerate(idx)
                b[i] += wqq * φ[li] * Vx * F0(k * x)
                for (lj, j) in enumerate(idx)
                    V[i, j] += wqq * φ[li] * Vx * φ[lj]
                end
            end
        end
    end
    A = E .* S .+ K .- V                                  # (E − H) weak form
    # BC: drop value DOF at r=0 (node 1) and value+slope[+curv] at rmax (node nint+1)
    dropped = [1, nint * ncol + 1]; ncol == 3 && push!(dropped, nint * ncol + 2)
    keep = setdiff(1:nd, dropped)
    c_keep = A[keep, keep] \ b[keep]
    c = zeros(ComplexF64, nd); c[keep] = c_keep
    resid = norm(A[keep, keep] * c_keep - b[keep]) / norm(b[keep])
    # amplitude: bra = regular F0, ket = u_tot, ∫·q dr
    M = 0.0im
    for ip in 1:nint
        a = mesh.knots[ip]; h = mesh.knots[ip + 1] - mesh.knots[ip]
        for g in 1:nq
            r = a + 0.5h * (uq[g] + 1.0); w = 0.5h * wq[g]
            x = contour_x(cs, r); q = contour_q(cs, r)
            idx, φ, _, _ = spline_functions(mesh, r)
            u = F0(k * x) + sum(c[idx[li]] * φ[li] for li in eachindex(idx))
            M += w * q * F0(k * x) * mt_1S0(x) * u
        end
    end
    f = -(1.0 / E) * M; Sm = 1.0 + 2im * k * f
    δ = rad2deg(0.5 * angle(Sm)); δ = δ < 0 ? δ + 180 : δ
    return δ, abs(Sm), resid
end

println("="^70)
println(" Galerkin spline + SMOOTH-ECS, MT ¹S₀ 2-body (target δ≈63.5°, η→1)")
println(" R0=6, w=2, quintic; θ-independence is the smooth-ECS signature")
println("="^70)
dom = [(0.0, 6.0, 30, 1.0), (6.0, 100.0, 120, 1.0)]   # knot on R0=6
@printf(" %4s | %10s %10s %10s\n", "θ°", "δ(deg)", "η", "resid")
for θd in (8.0, 10.0, 12.0, 14.0, 16.0)
    cs = CSContour(:smooth; θ_deg = θd, R0 = 6.0, w = 2.0)
    δ, η, res = solve_galerkin_2body(cs, 1.0, dom)
    @printf(" %4.0f | %10.3f %10.4f %10.2e\n", θd, δ, η, res)
end
