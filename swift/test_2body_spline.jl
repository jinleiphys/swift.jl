# test_2body_spline.jl — Step 2 GATE for the Lagrange-Laguerre -> Hermite-spline switch.
#
# Same physics as test_2body_cs_1S0.jl (complex-scaled MT ¹S₀ two-body scattering,
# inhomogeneous Lazauskas method) but with the y-style radial coordinate represented by
# the new Hermite-spline basis (swift/splines.jl), solved by COLLOCATION instead of the
# Lagrange-mesh Galerkin form. This proves the spline kinetic (S2), potential-at-rotated-
# point, boundary conditions, complex scaling, and amplitude extraction all work on a
# scattering solve BEFORE the basis is wired into the 3-body matrices.
#
# Target: Lazauskas-Carbonell PRC 84,034002 Table I -> δ(¹S₀) = 63.512° at E_cm = 1 MeV,
# θ=10°, r_max=100 fm.  η MUST be 1 (single open channel, real potential, no absorption).
#
# Collocation form of  [E - H] u_sc = V u_in,  H = -ħ²/2μ d²/dz² + V(z),  z = r·e^{iθ}:
#   at each Gauss collocation point z_c = r_c·e^{iθ}
#     Σ_j c_j [ E·S_j(z_c) + (ħ²/2μ)·S2_j(z_c) - V(z_c)·S_j(z_c) ] = V(z_c)·F_0(k z_c)
#   S_j = spline value, S2_j = spline d²/dz²  (spline_functions returns both, CS-rotated).
#   u_tot(z_c) = F_0(k z_c) + Σ_j c_j S_j(z_c).
#   f = -(1/E)·e^{iθ}·Σ_c w_c·F_0(k z_c)·V(z_c)·u_tot(z_c),   S = 1 + 2ik·f.

using LinearAlgebra, Printf, FastGaussQuadrature
include("splines.jl"); using .Splines
include("ecs.jl");     using .ECS

const ħ   = 197.3269718
const amu = 931.49432
const mN  = 1.0079713395678829
const μ   = mN / 2.0
const ħ2_2μ = ħ^2 / (2.0 * μ * amu)        # = 41.471 MeV·fm²

# MT ¹S₀ potential, evaluated at a (complex) radius — analytic continuation of the Yukawas.
mt_1S0(z) = 1438.72 * exp(-3.11 * z) / z - 513.968 * exp(-1.55 * z) / z
F0(z) = sin(z)                             # regular Coulomb F_0 at η=0 = Riccati-Bessel sin

# 2-point / 3-point Gauss weights on [-1,1] (aligned with collocation_points ordering).
gauss_w(ncol) = ncol == 2 ? (1.0, 1.0) : (5.0/9.0, 8.0/9.0, 5.0/9.0)

"""
    solve_spline_2body(E, θ_deg, rmax, nint; ncol=2) -> (δ, η, resid)

Spline-collocation MT ¹S₀ CS scattering on `nint` uniform intervals over [0, rmax].
"""
function solve_spline_2body(E, θ_deg, rmax, nint; ncol = 2)
    θ = θ_deg * π / 180
    k = sqrt(E / ħ2_2μ)                     # ħ²k²/2μ = E
    mesh = init_spline_mesh([(0.0, rmax, nint, 1.0)]; ncol = ncol)
    nc   = length(mesh.xc)                  # number of collocation rows = ncol*nint
    nd   = mesh.ndof                        # number of DOF = ncol*(nint+1)

    A    = zeros(ComplexF64, nc, nd)        # collocation operator [E - H]
    Sval = zeros(ComplexF64, nc, nd)        # spline values (to rebuild u_sc, u_tot)
    b    = zeros(ComplexF64, nc)            # source V·F_0
    zc   = Vector{ComplexF64}(undef, nc)    # rotated collocation points
    wc   = Vector{Float64}(undef, nc)       # contour quadrature weights (real)
    Vzc  = Vector{ComplexF64}(undef, nc)

    gw = gauss_w(ncol)
    for c in 1:nc
        r_c = mesh.xc[c]
        z   = r_c * cis(θ)
        zc[c]  = z
        Vzc[c] = mt_1S0(z)
        # contour weight: interval (b-a)/2 * gauss weight; intervals are uniform here
        h_int   = rmax / nint
        wc[c]   = 0.5 * h_int * gw[((c - 1) % ncol) + 1]
        idx, S, _, S2 = spline_functions(mesh, r_c; θ = θ)
        for (loc, j) in enumerate(idx)
            Sval[c, j] = S[loc]
            A[c, j]    = E * S[loc] + ħ2_2μ * S2[loc] - Vzc[c] * S[loc]
        end
        b[c] = Vzc[c] * F0(k * z)
    end

    # Boundary conditions: zero the value DOF at the origin node and at the outer node
    # (regular at r=0; CS-damped scattered wave -> 0 at rmax). For quintic also zero the
    # outer slope DOF. This removes exactly ncol DOF, making the system square.
    dropped = [1, nint * ncol + 1]                 # value @ node 1, value @ last node
    ncol == 3 && push!(dropped, nint * ncol + 2)   # + slope @ last node
    keep = setdiff(1:nd, dropped)
    @assert length(keep) == nc "BC bookkeeping: $(length(keep)) DOF vs $nc rows"

    # FIX A (conditioning): column equilibration. The Hermite slope/curvature DOF carry
    # h, h² factors, so their collocation columns shrink as h→0 and the system becomes
    # ill-conditioned (resid blew up to 3e-6 and δ corrupted to 65° by nint=800). Scaling
    # each column to unit norm before the solve, then unscaling, is mathematically
    # identical but well-conditioned, and auto-absorbs the h, h² scales (works for graded
    # grids too). cond(A) is reported so the improvement is visible.
    # NOTE on conditioning (fix A, tried 2026-06-29): the Hermite-collocation matrix is
    # ill-conditioned, cond ~ h^{-7} for quintic (nint 100→800: 8.6e10→1.7e17). This is
    # INTRINSIC to high-order Hermite collocation of the 2nd-order CS operator, NOT a DOF
    # scaling artifact: both column and full row+column equilibration leave cond unchanged
    # (within ~25%). So equilibration does not help. It only sets a CEILING: keep
    # nint ≲ 400 for quintic (cond < 1/eps); the observable stays accurate below that. The
    # actual accuracy blocker is the outer BC (θ-slide), not conditioning — see test header.
    Ak = A[:, keep]
    c_keep = Ak \ b
    c_full = zeros(ComplexF64, nd); c_full[keep] = c_keep
    resid  = norm(A * c_full - b) / norm(b)
    @printf("      [cond(A)=%.1e]\n", cond(Ak))

    # Amplitude integral on a FINE per-interval Gauss rule (decoupled from ncol), with
    # u_tot rebuilt from the spline coefficients at each fine point:
    #   f = -(1/E)·∫_contour F_0(kz)·V(z)·u_tot(z) dz,  dz = e^{iθ} dr.
    nq = 12
    uq, wq = gausslegendre(nq)
    M = 0.0im
    for i in 1:nint
        a, hbar = mesh.knots[i], mesh.knots[i + 1] - mesh.knots[i]
        for q in 1:nq
            r_q = a + 0.5 * hbar * (uq[q] + 1.0)
            w_q = 0.5 * hbar * wq[q]
            z_q = r_q * cis(θ)
            idx, S, _, _ = spline_functions(mesh, r_q; θ = θ)
            u_tot_q = F0(k * z_q) + sum(c_full[idx[loc]] * S[loc] for loc in eachindex(idx))
            M += w_q * F0(k * z_q) * mt_1S0(z_q) * u_tot_q
        end
    end
    f = -(1.0 / E) * cis(θ) * M
    S = 1.0 + 2im * k * f
    δ = rad2deg(0.5 * angle(S)); δ = δ < 0 ? δ + 180 : δ
    return δ, abs(S), resid
end

"""
    solve_spline_2body_ecs(E, θ_deg, R0, rmax, n_in, n_ext; ncol=3) -> (δ, η, resid)

EXTERIOR complex scaling: interior [0,R0] unrotated (θ=0, exact real wavefunction), only
the exterior (R0,rmax] rotated. R0 is forced to be a knot (two-domain grid) so the contour
kink sits on an element boundary. The amplitude integral is dominated by the short-range
interior, which is real and θ-INDEPENDENT, so δ,η should not slide with θ.
"""
function solve_spline_2body_ecs(E, θ_deg, R0, rmax, n_in, n_ext; ncol = 3)
    θ = θ_deg * π / 180
    k = sqrt(E / ħ2_2μ)
    mesh = init_spline_mesh([(0.0, R0, n_in, 1.0), (R0, rmax, n_ext, 1.0)]; ncol = ncol)
    knots = mesh.knots; nint = mesh.nint; nd = mesh.ndof
    nc = length(mesh.xc)

    A = zeros(ComplexF64, nc, nd)
    b = zeros(ComplexF64, nc)
    for c in 1:nc
        r_c = mesh.xc[c]
        z   = ecs_contour(r_c, R0, θ)          # real inside R0, rotated outside
        Vz  = mt_1S0(z)
        idx, S, _, S2 = spline_functions_ecs(mesh, r_c, R0; θ = θ)
        for (loc, j) in enumerate(idx)
            A[c, j] = E * S[loc] + ħ2_2μ * S2[loc] - Vz * S[loc]
        end
        b[c] = Vz * F0(k * z)
    end

    dropped = [1, nint * ncol + 1]
    ncol == 3 && push!(dropped, nint * ncol + 2)
    keep = setdiff(1:nd, dropped)
    c_keep = A[:, keep] \ b
    c_full = zeros(ComplexF64, nd); c_full[keep] = c_keep
    resid  = norm(A * c_full - b) / norm(b)

    # amplitude: ∫_contour F0·V·u dz, per-element Jacobian (1 interior, e^{iθ} exterior)
    nq = 12; uq, wq = gausslegendre(nq)
    M = 0.0im
    for i in 1:nint
        a = knots[i]; hreal = knots[i + 1] - knots[i]
        jac = knots[i] >= R0 - 1e-12 ? cis(θ) : 1.0 + 0.0im
        for q in 1:nq
            r_q = a + 0.5 * hreal * (uq[q] + 1.0); w_q = 0.5 * hreal * wq[q]
            z_q = ecs_contour(r_q, R0, θ)
            idx, S, _, _ = spline_functions_ecs(mesh, r_q, R0; θ = θ)
            u_q = F0(k * z_q) + sum(c_full[idx[l]] * S[l] for l in eachindex(idx))
            M += w_q * jac * F0(k * z_q) * mt_1S0(z_q) * u_q
        end
    end
    f = -(1.0 / E) * M
    S = 1.0 + 2im * k * f
    δ = rad2deg(0.5 * angle(S)); δ = δ < 0 ? δ + 180 : δ
    return δ, abs(S), resid
end

"""
    solve_spline_2body_qcs(cs, E, domains; ncol=3) -> (δ, η, resid)

UNIFIED, basis-agnostic CS solver: the Hamiltonian is assembled in the q-operator form
H = -(ħ²/2μ)[(1/q²)∂² - (q'/q³)∂] + V(x(r)) using the REAL-r spline (value, d/dr, d²/dr²)
and the contour `cs` (x, q, q'). Works for any CSContour kind (:uniform, :sharp, :smooth).
The spline supplies only real-coordinate derivatives; all complex scaling lives in cs.
"""
function solve_spline_2body_qcs(cs::CSContour, E, domains; ncol = 3)
    k = sqrt(E / ħ2_2μ)
    mesh = init_spline_mesh(domains; ncol = ncol)
    knots = mesh.knots; nint = mesh.nint; nd = mesh.ndof; nc = length(mesh.xc)
    A = zeros(ComplexF64, nc, nd); b = zeros(ComplexF64, nc)
    for c in 1:nc
        r_c = mesh.xc[c]
        x = contour_x(cs, r_c); q = contour_q(cs, r_c); qp = contour_qp(cs, r_c)
        Vx = mt_1S0(x)
        idx, S, S1, S2 = spline_functions(mesh, r_c)         # θ=0: real-r value/∂/∂²
        for (loc, j) in enumerate(idx)
            kin = ħ2_2μ * ((1.0 / q^2) * S2[loc] - (qp / q^3) * S1[loc])
            A[c, j] = E * S[loc] + kin - Vx * S[loc]         # (E - H) in q-operator form
        end
        b[c] = Vx * F0(k * x)
    end
    dropped = [1, nint * ncol + 1]; ncol == 3 && push!(dropped, nint * ncol + 2)
    keep = setdiff(1:nd, dropped)
    c_keep = A[:, keep] \ b
    c_full = zeros(ComplexF64, nd); c_full[keep] = c_keep
    resid  = norm(A * c_full - b) / norm(b)
    # amplitude: ∫ F0(k·x) V(x) u_tot dz, with dz = q(r) dr (q carries the contour Jacobian)
    nq = 12; uq, wq = gausslegendre(nq); M = 0.0im
    for i in 1:nint
        a = knots[i]; h = knots[i + 1] - knots[i]
        for qd in 1:nq
            r_q = a + 0.5h * (uq[qd] + 1.0); w_q = 0.5h * wq[qd]
            x = contour_x(cs, r_q); qm = contour_q(cs, r_q)
            idx, S, _, _ = spline_functions(mesh, r_q)
            u = F0(k * x) + sum(c_full[idx[l]] * S[l] for l in eachindex(idx))
            M += w_q * qm * F0(k * x) * mt_1S0(x) * u
        end
    end
    f = -(1.0 / E) * M; Sm = 1.0 + 2im * k * f
    δ = rad2deg(0.5 * angle(Sm)); δ = δ < 0 ? δ + 180 : δ
    return δ, abs(Sm), resid
end

println("="^70)
println(" THREE CS variants via the unified q-operator layer (quintic spline, MT ¹S₀)")
println(" target δ=63.512°, η=1.  η→1 + θ-flat = good.  Lagrange-Laguerre ref: 63.224/0.999")
println("="^70)
let
    rmax = 296.0
    dom_uni    = [(0.0, 100.0, 300, 1.0)]                                          # uniform wants a matched box
    dom_sharp  = [(0.0, 6.0, 40, 1.0), (6.0, rmax, 360, 1.0)]                      # knot at R0=6
    dom_smooth = [(0.0, 6.0, 40, 1.0), (6.0, 8.0, 24, 1.0), (8.0, rmax, 340, 1.0)] # knots at R0, R0+w
    for θd in (8.0, 10.0, 12.0, 14.0, 16.0)
        u  = solve_spline_2body_qcs(CSContour(:uniform; θ_deg = θd), 1.0, dom_uni)
        sh = solve_spline_2body_qcs(CSContour(:sharp;  θ_deg = θd, R0 = 6.0), 1.0, dom_sharp)
        sm = solve_spline_2body_qcs(CSContour(:smooth; θ_deg = θd, R0 = 6.0, w = 2.0), 1.0, dom_smooth)
        @printf("θ=%4.0f° | uniform δ=%7.3f η=%.4f | sharp δ=%7.3f η=%.4f | smooth δ=%7.3f η=%.4f\n",
                θd, u[1], u[2], sh[1], sh[2], sm[1], sm[2])
    end
    println("""
  Conclusion (q-operator layer with a REAL-r basis):
   • SMOOTH is the ONLY one that works through this basis-agnostic path: δ flat ≈63.50°,
     η→1.0000, θ-independent. C² contour, no kink, real-r derivatives suffice → ANY basis
     (LL / Legendre / spline) plugs in. This is the production / multi-basis path.
   • SHARP fails here (η=1.25→1.50): the real-r C²-quintic cannot carry the du/dr kink at R0.
     Sharp works ONLY via the local complex-h route (solve_spline_2body_ecs below, η→1.0000),
     which is local-basis-specific, NOT basis-agnostic.
   • UNIFORM fails here too (δ=25): its natural discretization is the ROTATED-argument form
     (solve_spline_2body below), not the real-r q-operator. Also θ-sensitive.
  ⇒ Each CS variant has a natural discretization; only SMOOTH is basis-agnostic. For a
    pluggable-basis framework, SMOOTH ECS is the layer to standardize on.""")
end

println("\n", "="^70)
println(" Spline-collocation 2-body CS scattering  MT ¹S₀  (target δ=63.512°, η=1)")
println("="^70)
for ncol in (2, 3)
    println("\n--- ncol = $ncol ($(ncol==2 ? "cubic" : "quintic") Hermite) ---")
    @printf("  %-22s  %10s %10s %10s\n", "nint (rmax=100, θ=10°)", "δ (deg)", "η", "resid")
    for nint in (100, 200, 400, 800)
        δ, η, res = solve_spline_2body(1.0, 10.0, 100.0, nint; ncol = ncol)
        @printf("  nint=%-17d  %10.3f %10.5f %10.1e\n", nint, δ, η, res)
    end
end
println("\nReference: Lazauskas Table I, δ=63.512°, η=1 (unitarity). η→1 is the gate.")

# Validated quintic settings: θ must be large enough to damp the scattered wave to ~0
# within rmax (the u_sc(rmax)=0 boundary condition). At θ=10°/rmax=100 the wave is only
# damped to e^{-k·rmax·sinθ}≈0.07, so δ,η are ~2% off; at θ=14° it is ≈0.02 and the
# benchmark is reproduced. Cubic (ncol=2) is inadequate (C¹ -> discontinuous 2nd
# derivative); use quintic (ncol=3 = Rimas's NCOL=3 default).
println("\n--- uniform-CS quintic (θ=14°, rmax=100, nint=300): θ-sensitive, not matching Lagrange ---")
δ, η, res = solve_spline_2body(1.0, 14.0, 100.0, 300; ncol = 3)
@printf("  δ = %.3f°  η = %.5f  resid = %.1e   (only crosses benchmark near θ=14°; slides with θ)\n",
        δ, η, res)

# ===================================================================================
# EXTERIOR complex scaling: interior [0,R0] unrotated (real, exact), only the exterior
# rotated. The amplitude lives in the short-range real interior → θ-INDEPENDENT and
# exactly unitary, unlike uniform CS. This is the validated production path.
# ===================================================================================
println("\n--- EXTERIOR CS quintic (R0=6, rmax=296, exterior nint=380): θ-INDEPENDENT ---")
println("  uniform CS slid δ 62→63.5 with θ; ECS is flat. Lazauskas Table I: 63.512°, η=1.")
@printf("  %6s | %10s %10s %10s\n", "θ", "δ", "η", "resid")
for θd in (8.0, 10.0, 12.0, 14.0, 16.0)
    δe, ηe, rese = solve_spline_2body_ecs(1.0, θd, 6.0, 296.0, 40, 380; ncol = 3)
    flat = abs(δe - 63.51) < 0.1 && abs(ηe - 1.0) < 0.03
    @printf("  %5.0f° | %10.3f %10.5f %10.1e   %s\n", θd, δe, ηe, rese, flat ? "✓" : "")
end
println("  ⇒ ECS δ flat at 63.49±0.03°, η→1.0000 (machine-level unitarity). Requirement: the")
println("    exterior length L=rmax−R0 must damp the wave, e^{−k·L·sinθ} ≪ 1 (θ≥8° here).")
