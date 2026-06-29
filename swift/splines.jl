module Splines
export SplineMesh,
       init_spline_mesh,
       graded_grid,
       spline_functions,
       collocation_points,
       find_interval
# Hermite finite-element ("spline") basis for the radial coordinate, ported from
# R. Lazauskas' Faddeev code (Ex-Fad_eq / module SPLINES). This is the local,
# finite-interval alternative to the Lagrange-Laguerre basis in laguerre.jl:
#
#   Lagrange-Laguerre (laguerre.jl) : global functions on [0,inf), points cluster
#                                     at the origin, non-orthonormal 1/sqrt(y) tail.
#   Hermite spline    (this file)   : piecewise polynomials on a finite knot grid
#                                     [0, rmax], LOCAL support (only 2*ncol functions
#                                     nonzero at any point), banded overlap, no tail.
#
# Two element orders are supported, following Lazauskas' NCOL switch:
#   ncol = 2 : cubic  Hermite, 2 DOF per node (value, slope)            -> C^1
#   ncol = 3 : quintic Hermite, 3 DOF per node (value, slope, curvature)-> C^2
#
# Complex scaling enters exactly as in swift's back-rotation convention: the knots
# stay real and the evaluation argument is rotated, r -> r*exp(i*theta) (cf. SPL_CMP).


# ----------------------------------------------------------------------------------
# Hermite shape functions on the reference interval u in [0,1].
# Naming: <basis>(u) = value, <basis>_p(u) = d/du, <basis>_s(u) = d^2/du^2.
# ----------------------------------------------------------------------------------

# Cubic Hermite (ncol = 2):
#   hv : carries the nodal VALUE  (hv(0)=1, hv(1)=0, hv'(0)=hv'(1)=0)
#   hs : carries the nodal SLOPE  (hs(0)=0, hs'(0)=1, hs(1)=hs'(1)=0)
hv(u)   = 1.0 - 3.0u^2 + 2.0u^3
hv_p(u) = -6.0u * (1.0 - u)
hv_s(u) = -6.0 * (1.0 - 2.0u)
hs(u)   = u * (1.0 - u)^2
hs_p(u) = (1.0 - u) * (1.0 - 3.0u)
hs_s(u) = 2.0 * (3.0u - 2.0)

# Quintic Hermite (ncol = 3):
#   q1 : nodal VALUE,  q2 : nodal SLOPE,  q3 : nodal CURVATURE
q1(u)   = u^3 * (3.0 * (u - 1.0) * (2.0 * (u - 1.0) - 1.0) + 1.0)
q1_p(u) = 30.0u^2 * (u - 1.0)^2
q1_s(u) = 60.0u * (u - 1.0) * (2.0u - 1.0)
q2(u)   = u^3 * (u - 1.0) * (4.0 - 3.0u)
q2_p(u) = u^2 * (-15.0u^2 + 28.0u - 12.0)
q2_s(u) = 12.0u * (-5.0u^2 + 7.0u - 2.0)
q3(u)   = 0.5u^3 * (u - 1.0)^2
q3_p(u) = 0.5u^2 * (u - 1.0) * (5.0u - 3.0)
q3_s(u) = u * (10.0u^2 - 12.0u + 3.0)


# ----------------------------------------------------------------------------------
# Graded knot grid (port of G1D / G2D / G3D / GND).
# ----------------------------------------------------------------------------------

"""
    graded_grid(domains) -> Vector{Float64}

Build a monotonically increasing knot vector from one or more domains, each with its
own geometric step ratio. This is Lazauskas' G1D/G2D/G3D mesh: useful for putting
fine knots where the wavefunction oscillates and coarse knots where it is smooth.

# Arguments
- `domains`: vector of `(x0, xN, n, ratio)` tuples. Each domain contributes `n`
  sub-intervals spanning `[x0, xN]`. `ratio = 1.0` gives a uniform step; `ratio > 1`
  grows the step geometrically (coarser toward `xN`), `ratio < 1` shrinks it.
  Domains must be contiguous (`xN` of one equals `x0` of the next).

# Returns
- Knot vector `x[0..N]` (length `sum(n)+1`), `x[1] = domains[1].x0`,
  `x[end] = domains[end].xN`.

# Example
```julia
# fine near origin, coarser outward, single domain [0,60] with 80 uniform intervals
knots = graded_grid([(0.0, 60.0, 80, 1.0)])
# two domains: dense inner region, geometric outer region
knots = graded_grid([(0.0, 20.0, 40, 1.0), (20.0, 60.0, 30, 1.05)])
```
"""
function graded_grid(domains::AbstractVector)
    n_total = sum(d[3] for d in domains)
    x = Vector{Float64}(undef, n_total + 1)   # 1-based: x[1]..x[n_total+1]

    x[1] = domains[1][1]
    k = 1                                      # index of the last filled knot
    for (x0, xN, n, ratio) in domains
        @assert isapprox(x[k], x0; atol = 1e-12) "graded_grid: domains must be contiguous"
        # First step length so that the geometric sum of n steps spans [x0, xN].
        dx = ratio == 1.0 ? (xN - x0) / n : (xN - x0) * (ratio - 1.0) / (ratio^n - 1.0)
        for _ in 1:n
            x[k + 1] = x[k] + dx
            k += 1
            dx *= ratio
        end
        x[k] = xN                              # pin the right edge exactly
    end
    return x
end


# ----------------------------------------------------------------------------------
# Mesh container (mirrors mesh.jl's meshset / initialmesh pattern).
# ----------------------------------------------------------------------------------

"""
    SplineMesh

Hermite-spline mesh for one radial coordinate.

# Fields
- `knots`     : knot vector `x[1..N+1]` (N = number of intervals).
- `ncol`      : element order / DOF-per-node (2 = cubic, 3 = quintic).
- `nint`      : number of intervals `N`.
- `ndof`      : raw number of Hermite DOF before boundary conditions, `ncol*(N+1)`.
- `xc`        : collocation points (`ncol` Gauss points per interval, length `ncol*N`).
- `rmax`      : outer boundary `knots[end]`.

The raw DOF are ordered node by node: node `j` (j = 1..N+1) owns the `ncol` DOF
`(j-1)*ncol + 1 : j*ncol` = (value, slope, [curvature]). Boundary conditions
(regular at the origin, outer condition at `rmax`) are applied by the operator
assembly that consumes this mesh, not here, so the basis stays a faithful port.
"""
struct SplineMesh
    knots::Vector{Float64}
    ncol::Int
    nint::Int
    ndof::Int
    xc::Vector{Float64}
    rmax::Float64
end

"""
    init_spline_mesh(domains; ncol=3) -> SplineMesh

Construct a `SplineMesh` from a graded-grid domain specification.

# Arguments
- `domains`: domain spec passed straight to [`graded_grid`](@ref).
- `ncol`   : 2 (cubic Hermite) or 3 (quintic Hermite).

# Example
```julia
mesh = init_spline_mesh([(0.0, 60.0, 80, 1.0)]; ncol = 3)   # 80 quintic elements on [0,60]
```
"""
function init_spline_mesh(domains::AbstractVector; ncol::Int = 3)
    @assert ncol in (2, 3) "SplineMesh: ncol must be 2 (cubic) or 3 (quintic)"
    knots = graded_grid(domains)
    nint  = length(knots) - 1
    ndof  = ncol * (nint + 1)
    xc    = collocation_points(knots, ncol)
    return SplineMesh(knots, ncol, nint, ndof, xc, knots[end])
end


# ----------------------------------------------------------------------------------
# Interval search (port of SPL's binary search / the `interval` function).
# ----------------------------------------------------------------------------------

"""
    find_interval(knots, r) -> Int

Return the index `ip` such that `knots[ip-1] < r <= knots[ip]` (1-based, so `r` lies
in the `ip`-th interval `[knots[ip], knots[ip+1]]` in 0-based knot terms). Binary
search, matching Lazauskas' SPL. `r` must lie in `[knots[1], knots[end]]`.
"""
function find_interval(knots::AbstractVector{<:Real}, r::Real)
    @assert knots[1] <= r <= knots[end] "find_interval: r=$r outside [$(knots[1]), $(knots[end])]"
    lo, hi = 1, length(knots)
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if r > knots[mid]
            lo = mid
        else
            hi = mid
        end
    end
    return hi
end


# ----------------------------------------------------------------------------------
# Collocation points (port of COLLOC): ncol Gauss points per interval.
# ----------------------------------------------------------------------------------

"""
    collocation_points(knots, ncol) -> Vector{Float64}

Return the `ncol` Gauss-Legendre collocation abscissae in each of the `N` intervals
defined by `knots` (total length `ncol*N`), ordered interval by interval. Matches
Lazauskas' COLLOC: 2-point Gauss for `ncol=2`, 3-point Gauss for `ncol=3`.
"""
function collocation_points(knots::AbstractVector{<:Real}, ncol::Int)
    # Gauss-Legendre nodes on [-1,1] for the two supported orders (hard-coded to match
    # COLLOC exactly; FastGaussQuadrature would give the same values).
    nodes = ncol == 2 ? (-0.5773502691896257, 0.5773502691896257) :
                        (-0.7745966692414834, 0.0, 0.7745966692414834)
    nint = length(knots) - 1
    xc = Vector{Float64}(undef, ncol * nint)
    ig = 1
    for i in 1:nint
        a, b = knots[i], knots[i + 1]
        mid, half = 0.5 * (b + a), 0.5 * (b - a)
        for u in nodes
            xc[ig] = mid + half * u
            ig += 1
        end
    end
    return xc
end


# ----------------------------------------------------------------------------------
# Basis evaluation (port of SPL / SPL_CMP).
# ----------------------------------------------------------------------------------

"""
    spline_functions(mesh, r; θ=0.0) -> (idx, S, S1, S2)

Evaluate the Hermite basis at point `r`. Only the `2*ncol` functions supported on the
interval containing `r` are nonzero, so this returns just those, together with their
global DOF indices.

# Arguments
- `mesh` : a [`SplineMesh`](@ref).
- `r`    : real evaluation point in `[knots[1], knots[end]]`.
- `θ`    : complex-scaling angle in radians. The knots stay real; the argument is
           rotated `r -> r*exp(iθ)` (Lazauskas' SPL_CMP convention, which is swift's
           back-rotation). `θ=0` reduces to the real SPL.

# Returns
- `idx::Vector{Int}`        : global DOF indices of the `2*ncol` local functions.
- `S, S1, S2`               : the function values and their 1st / 2nd derivatives with
                              respect to the (real) knot coordinate at `r`. Real for
                              `θ=0`, `ComplexF64` otherwise. The kinetic builder applies
                              the usual `exp(-2iθ)` scaling, consistent with T_matrix.

The local ordering is `[left-node DOF (value, slope[, curvature]),
right-node DOF (value, slope[, curvature])]`, matching SPL's S(1..2*ncol).
"""
function spline_functions(mesh::SplineMesh, r::Real; θ::Real = 0.0)
    knots = mesh.knots
    ncol  = mesh.ncol
    ip    = find_interval(knots, r)            # r in [knots[ip-1], knots[ip]]

    h  = knots[ip] - knots[ip - 1]             # interval length (real, as in SPL_CMP)
    Rp = 1.0 / h
    is_cs = (θ != 0.0)
    # Rotated local coordinate R = (r*e^{iθ} - x_left)/h in [0,1] (or complex under CS).
    dx = is_cs ? (r * cis(θ) - knots[ip - 1]) : (r - knots[ip - 1])
    R  = dx / h

    T  = is_cs ? ComplexF64 : Float64
    S  = Vector{T}(undef, 2ncol)
    S1 = Vector{T}(undef, 2ncol)
    S2 = Vector{T}(undef, 2ncol)

    if ncol == 2
        # left node (value, slope), then right node (value, slope)
        S[1]  = hv(R);            S1[1] = hv_p(R) * Rp;       S2[1] = hv_s(R) * Rp^2
        S[2]  = hs(R) * h;        S1[2] = hs_p(R);            S2[2] = hs_s(R) * Rp
        S[3]  = hv(1.0 - R);      S1[3] = -hv_p(1.0 - R) * Rp; S2[3] = hv_s(1.0 - R) * Rp^2
        S[4]  = -hs(1.0 - R) * h; S1[4] = hs_p(1.0 - R);      S2[4] = -hs_s(1.0 - R) * Rp
    else # ncol == 3
        # left node (value, slope, curvature)
        S[1]  = q1(1.0 - R);        S1[1] = -q1_p(1.0 - R) * Rp;   S2[1] = q1_s(1.0 - R) * Rp^2
        S[2]  = -q2(1.0 - R) * h;   S1[2] = q2_p(1.0 - R);         S2[2] = -q2_s(1.0 - R) * Rp
        S[3]  = q3(1.0 - R) * h^2;  S1[3] = -q3_p(1.0 - R) * h;    S2[3] = q3_s(1.0 - R)
        # right node (value, slope, curvature)
        S[4]  = q1(R);              S1[4] = q1_p(R) * Rp;          S2[4] = q1_s(R) * Rp^2
        S[5]  = q2(R) * h;          S1[5] = q2_p(R);               S2[5] = q2_s(R) * Rp
        S[6]  = q3(R) * h^2;        S1[6] = q3_p(R) * h;           S2[6] = q3_s(R)
    end

    # Global DOF indices: the two nodes (ip-1, ip) own consecutive DOF blocks.
    base = (ip - 2) * ncol         # 0-based offset of the left node's first DOF
    idx  = collect(base + 1 : base + 2ncol)

    return idx, S, S1, S2
end

end # module Splines
