# swift.jl devlog (dead-ends, abandoned paths)

Reverse-chronological. Auto-loaded context, so keep it to lessons future sessions must not relitigate.
Completed work lives in TODO.md; forward-looking decisions live in CLAUDE.md. This file is failures only.

---

## 2026-06-30: the spline y-kinetic must NOT use the real-r ×e^{-2iθ} form under uniform CS (θ-dependent error)

**Why we tried it:** for the 3-body uniform-CS scattering I built the spline y-overlap/kinetic as REAL Galerkin
matrices (Sy_real, Ky_real) and applied CS by multiplying the kinetic by e^{-2iθ} (and leaving Sy real),
mirroring the LL code's "real-r + e^{-2iθ}" scheme. Reproduced doublet-14.1 at θ=3° (matched LL term-by-term).
**What failed:** at θ=6° the spline gives η=0.69 while the LL reference gives η=0.43 at the same angle (both
slide δ with the box, but LL's η is θ-stable ≈0.43-0.47, mine is not). So a θ-DEPENDENT discrepancy: spline≡LL
at θ=3°, diverges by θ=6°. The δ does trend to the benchmark with box (GMRES balanced-box 50→70→90: δ=117→
113→108.5), but η is wrong at θ>3°.
**Root cause:** the real-r ×e^{-2iθ} scheme is WRONG for a spline. Under uniform CS the physical wave is
ψ(r·e^{iθ}); a REAL-r spline Σc_iφ_i(r) cannot represent that globally-rotated oscillatory function, so
scaling ∂² by e^{-2iθ} on a real-r basis is not the CS operator. The 2-body already proved this (solve_spline_2body_qcs
:uniform → δ=25, vs rotated-argument → 62, vs smooth-ECS q-operator → 63.5). The error scales with θ: tiny at
3° (masked by the x-box under-convergence), dominant by 6°. The LL analytic basis tolerates real-r ×e^{-2iθ}
(its functions carry the rotation); the spline does not.
**Lesson:** for uniform CS the spline y must use the ROTATED-ARGUMENT form — z=y·e^{iθ}, basis and S2 via
spline_functions(mesh,y;θ), with the e^{iθ} contour measure on Sy and the centrifugal at z². Crucially the Rxy
is then FEASIBLE without the smooth-ECS off-contour problem: under a uniform rotation the rearranged point
e^{iθ}·ξb_real lies on the contour, so spline_functions(mesh,ξb_real;θ) evaluates it correctly. Rebuild Sy, Ky,
Rxy, the F_λ source projection, and the amplitude all in rotated-argument; re-check θ-independence of η.
**Status:** Active fix (rotated-argument spline for uniform CS). The GMRES infrastructure + the C_n
normalization + the mixed-Rxy structure all stay; only the y CS bookkeeping changes.

## 2026-06-30: smooth-ECS on the 3-body Rxy is PARKED (hard, non-standard); use uniform CS + spline-y first

**Why we tried it:** decision A wanted smooth-ECS on both x and y (interior-real amplitude + breakup damping
in x). The 2-body validated cleanly. Plan was to extend the same contour to the 3-body rearrangement.
**What failed:** the 3-body Rxy under smooth ECS needs ψ at the rearranged coords πb=√(a²x²+b²y²+2ab·xy·cosθ_ang)
with x=x(rx), y=y(ry) on the contour → πb,ξb are COMPLEX and generally do NOT lie on the contour (sqrt of a
complex combination, not the contour's image). Evaluating a spline (or LL) basis at such an off-contour
complex point needs cross-element analytic continuation, which a piecewise polynomial does not support.
**Root cause:** a NON-uniform contour does not commute with the Jacobi rearrangement (a rotation in the (x,y)
plane). Uniform CS commutes (πb_phys = e^{iθ}·πb_real, homogeneous) → the existing real Rxy works unchanged;
smooth/exterior does not. Rimas's own benchmark uses UNIFORM CS + spline, not exterior — confirming exterior
CS on the 3-body rearrangement is non-standard. The 2-body had no Rxy, so smooth ECS was clean there.
**Lesson:** reproduce 14.1/42 first with uniform CS + spline-y (Rxy stays real, the validated θ=0 mixed Rxy
is reused verbatim; the spline only has to fix the LL 42-MeV mesh wall). Treat smooth-ECS-on-the-3-body as a
separate research refinement (off-contour evaluation) to attempt only after the uniform path reproduces the
benchmark. Uniform CS already rotates x, so breakup-along-x is still damped; smooth ECS's only extra gain is
interior-real amplitude, which Rimas's Green's-theorem extraction already handles under uniform CS.
**Status:** Parked (smooth-ECS 3-body Rxy). Active path: uniform CS + spline-y (Jin's call 2026-06-30).

## 2026-06-30: anisotropic complex scaling (θ_x ≠ θ_y) is incompatible with the real, angle-free Rxy

**Why we tried it:** to cheaply confirm "the breakup tail in x needs its own rotation" by turning OFF the x
rotation alone (θ_x=0, θ_y=θ) on the existing uniform-CS 3-body path and watching η degrade.
**What failed:** the test cannot isolate breakup-along-x at all; it would produce a meaningless number, so it
was discarded before running (caught by reading the code, not by a bad result).
**Root cause:** Rxy is built REAL and angle-free (`Rxy_matrix_optimized`, geometric coeffs a,b,c,d=−0.5,1,−0.75,−0.5).
Uniform CS keeps it real ONLY because a single common θ factors out of the homogeneous map x₃=a·x₁+b·y₁ → λ·x₃
(λ=e^{iθ}). With θ_x≠θ_y the phase no longer factors, so the real Rxy is the WRONG transform; mixing a real-x
potential with an Rxy that assumes a rotated x is inconsistent.
**Lesson:** never rotate x and y by different angles. Smooth-ECS with a COMMON θ confined to the exterior
(interior real) is the only Rxy-consistent generalization of uniform CS — that is decision A. To check
"breakup populates large x" instead, scan |ψ_sc(x)| at fixed y (`swift/test_breakup_xtail.jl`, which found
30-46% of the scattered ³S₁ x-norm beyond the deuteron range); it touches no Rxy.
**Status:** Abandoned (replaced by the shared-contour smooth-ECS, decision A in CLAUDE.md).

## 2026-06-29: sharp and uniform CS both fail through the basis-agnostic q-operator (only smooth works)

**Why we tried it:** to make the CS layer basis-agnostic (LL / Legendre / spline all plug in), assemble H
in the q-operator form −(ħ²/2μ)[(1/q²)∂²−(q'/q³)∂]+V(x(r)) with a REAL-r basis and select the contour kind
(:uniform / :sharp / :smooth). Goal: one code path for all three CS variants and all bases.
**What failed:** through the real-r q-operator path only :smooth reproduced the MT ¹S₀ benchmark (δ flat
63.50°, η→1.0000, θ-independent). :sharp gave η=1.25→1.50 (drifts up with θ); :uniform gave δ=25 (vs the
rotate-argument uniform's 62.2).
**Root cause:** sharp ECS has a du/dr kink at R0 that a C²-continuous real-r quintic cannot carry; uniform
CS's natural discretization is the rotated-argument form, not real-r. Both are basis-specific; only smooth
ECS (C² contour, no kink) is representable by a smooth real-r basis, hence basis-agnostic.
**Lesson:** standardize the multi-basis framework on SMOOTH ECS. Keep sharp only as the local complex-h
spline method (`solve_spline_2body_ecs`); do not expect sharp/uniform to work through the real-r q-operator.
**Status:** Replaced by smooth ECS as the basis-agnostic CS layer (`swift/ecs.jl`).

