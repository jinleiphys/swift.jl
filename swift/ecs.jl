module ECS
export CSContour, contour_x, contour_q, contour_qp
# Complex-scaling CONTOUR layer, basis-agnostic. Provides the contour point x(r), the
# metric q(r) = dx/dr, and q'(r) for three CS variants. The Hamiltonian is assembled in
# the q-operator form
#
#     H = -(ħ²/2μ) (1/q) d/dr [ (1/q) d/dr ] + V(x(r))
#       = -(ħ²/2μ) [ (1/q²) d²/dr² - (q'/q³) d/dr ] + V(x(r)),
#
# so the BASIS only has to supply real-coordinate values and real d/dr, d²/dr². Any basis
# (Lagrange-Laguerre, Lagrange-Legendre, Hermite spline, a neural net) plugs in unchanged;
# all the complex scaling lives in (x, q, q'). This is the same formulation as PINN-ECS.
#
# Three kinds (Jin's request: original / hard-cutoff / smooth):
#   :uniform  original uniform CS, whole axis rotated:
#               x = r·e^{iθ},  q = e^{iθ} (const),  q' = 0
#   :sharp     hard-cutoff exterior CS, kink at R0 (interior real):
#               x = r (r≤R0) | R0+(r-R0)e^{iθ} (r>R0),  q = 1 | e^{iθ},  q' = 0
#               NOTE: the solution has a du/dr kink at R0, so this needs a LOCAL basis with
#               a knot at R0 that does not enforce C¹ there; smooth/uniform do not.
#   :smooth    smooth exterior CS (PINN-ECS), C² contour, no kink:
#               x = r + (e^{iθ}-1)·I(r),  q = 1 + (e^{iθ}-1)·s(t),  q' = (e^{iθ}-1)·s'(t)/w
#               with t=(r-R0)/w and the cubic smoothstep s. Works with ANY basis.

struct CSContour
    kind::Symbol      # :uniform | :sharp | :smooth
    θ::Float64        # rotation angle [rad]
    R0::Float64       # rotation start [fm] (ignored for :uniform)
    w::Float64        # smooth transition width [fm] (only :smooth)
end
"""
    CSContour(kind; θ_deg=0.0, R0=0.0, w=1.0)

Build a complex-scaling contour. `kind ∈ (:uniform, :sharp, :smooth)`.
"""
CSContour(kind::Symbol; θ_deg::Real = 0.0, R0::Real = 0.0, w::Real = 1.0) =
    CSContour(kind, θ_deg * π / 180, Float64(R0), Float64(w))

# cubic smoothstep s(t) and its derivative on the normalized transition coordinate t
_s(t)  = t <= 0 ? 0.0 : t >= 1 ? 1.0 : 3t^2 - 2t^3
_sp(t) = (t <= 0 || t >= 1) ? 0.0 : 6t - 6t^2
# I(r) = ∫_{R0}^r s((t-R0)/w) dt  (closed form of the cubic smoothstep integral)
function _I(r, R0, w)
    r <= R0 && return 0.0
    t = (r - R0) / w
    t >= 1 ? 0.5w + (r - R0 - w) : w * (t^3 - 0.5 * t^4)
end

"Contour point x(r) (complex)."
function contour_x(c::CSContour, r::Real)
    e = cis(c.θ)
    c.kind === :uniform && return r * e
    c.kind === :sharp   && return r <= c.R0 ? complex(r) : c.R0 + (r - c.R0) * e
    return r + (e - 1) * _I(r, c.R0, c.w)              # :smooth
end

"Metric q(r) = dx/dr (complex)."
function contour_q(c::CSContour, r::Real)
    e = cis(c.θ)
    c.kind === :uniform && return e
    c.kind === :sharp   && return r <= c.R0 ? complex(1.0) : e
    return 1.0 + (e - 1) * _s((r - c.R0) / c.w)        # :smooth
end

"Metric derivative q'(r) (nonzero only for :smooth)."
function contour_qp(c::CSContour, r::Real)
    c.kind === :smooth || return 0.0 + 0.0im
    return (cis(c.θ) - 1) * _sp((r - c.R0) / c.w) / c.w
end

end # module ECS
