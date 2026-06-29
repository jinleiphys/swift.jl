# swift.jl devlog (dead-ends, abandoned paths)

Reverse-chronological. Auto-loaded context, so keep it to lessons future sessions must not relitigate.
Completed work lives in TODO.md; forward-looking decisions live in CLAUDE.md. This file is failures only.

---

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

## 2026-06-29: column/row equilibration does NOT fix the Hermite-collocation conditioning

**Why we tried it:** the Hermite spline collocation matrix is ill-conditioned (cond ~ h^{-7} for quintic;
resid grew 1e-10→3e-6 and δ corrupted to 65 by nint=800), blamed on the h, h² slope/curvature DOF scaling;
expected diagonal equilibration to cure it.
**What failed:** column AND full row+column equilibration left cond unchanged within ~25% (quintic nint=100:
8.6e10→7.0e10; nint=800: 1.7e17→1.1e17). Equilibration ineffective.
**Root cause:** the ill-conditioning is INTRINSIC to high-order Hermite collocation of the 2nd-order
operator, not a diagonal-scaling artifact, so no equilibration can touch it.
**Lesson:** do not chase it with equilibration; it only sets a nint ceiling (≲400 for quintic) and is
non-binding at usable resolution. The real accuracy blocker was the CS/BC choice (→ smooth ECS).
**Status:** Abandoned (equilibration); conditioning deprioritized as non-binding.

## 2026-06-29: "more points + larger θ" cannot make Lagrange-Laguerre converge doublet-42 δ

**Why we tried it:** after Rimas's 2nd email (θ=3° too small; use θ near the upper bound to damp the
outgoing wave fast) the natural fix was: at 42 MeV push θ up to ~7° and just add grid points + grow the
box until δ,η plateau on the benchmark (41.35°/0.5022).

**What failed:** density converges each box, but the converged value still slides with box size, with no
plateau. θ=7° square boxes: δ=53.9°(L=30) → 44.8°(L=55) → 28.8°(L=70); it crosses 41.35° only by accident
near L≈60, never flat. η does converge cleanly to ~0.50, but δ does not. At small θ=3° there IS a δ plateau,
but at the wrong value (~52°) unless xmax is also grown to ≈ymax (then it marches 52°→44°), and the LL
basis overflows (basis values ~1e150) past ny≈180 before you reach a balanced box big enough to finish.

**Root cause:** a genuine squeeze specific to the Lagrange-Laguerre mesh, not a formula/operator/prefactor
bug (all of those were verified component-by-component: prefactor 1/E_cm correct, ψsc exact to 1e-10 vs
dense A\b, F_λ accurate to 1e-15, T/B/V/Rxy validated on bound states). At small θ you need a huge BALANCED
(x,y) box because Rxy couples x↔y and the high-energy scattered wave reaches ~ymax in y and maps to large x;
LL overflows before you get there. At large θ the box can be small, but the CS incoming bra grows as
e^{+qy·sinθ} and contaminates the amplitude integral box-dependently → no plateau. LL clusters points near
the origin and covers the asymptotic region poorly, so it cannot represent the 42-MeV oscillatory scattered
wave (λ≈6.6 fm) on a small box. Rimas's benchmark used spline collocation (free point placement, tolerates
the asymmetric/large box without origin-clustering or overflow), which is why his LL-incapable regime is
exactly where swift stalls.

**Lesson:** do NOT keep tuning (θ, nx, ny, box) to force the LL-Laguerre y-mesh onto the 42-MeV benchmark.
η reproduces; δ is mesh-limited. The fix is the basis, not more points: switch the scattering coordinate y
from Lagrange-Laguerre (semi-infinite, origin-clustered) to Lagrange-Legendre on a finite box [0,ymax]
(uniform-ish coverage of the asymptotic region; swift's DBMM/bound-state side already has this
infrastructure). 14.1 MeV reproduces perfectly with LL because its y-extent is small; only high energy
exposes the mesh wall.

**Status:** Abandoned (LL + point-pushing). Replaced by the y→Lagrange-Legendre basis switch (see TODO.md).

---

## 2026-06-26: the "Lagrange-Laguerre overlap-tail" was NOT the root cause of doublet-42

**Why we tried it:** the 2026-06-17 diagnosis blamed the non-orthonormal overlap N[i,j]=δ+(-1)^{i-j}/√(yᵢyⱼ)
1/√y tail for coupling the growing CS incoming wave from small y to large y, making the amplitude integral
diverge with the box. The candidate cure was to orthogonalize the overlap or cut the basis tail.

**What failed:** cutting the basis tail (TAILCUT test) made EVERY point worse, including the working
doublet-14.1 (δ 105.95°→101.57°) and the quartet (η 1.386→1.532). The tail is legitimate and needed. Also,
a converged result must be basis-independent (Jin's objection), yet δ at fixed energy drifted with the LL
scaling parameter alpha (52.15→52.47) and strongly with xmax — proving the answer was simply NOT converged,
not corrupted by a tail.

**Root cause:** the real defect at 42 MeV is (x,y) coupled-box UNDER-CONVERGENCE. swift hardwired xmax=30 ≪
ymax=100, but Rxy couples x and y, so at high energy (large y-extent) the x-box must scale with the y-box.
Growing nx WITH xmax (keeping density) marches doublet-42 monotonically toward the benchmark:
δ=52.2°(xmax30)→49.6→47.7→46.3→45.4→44.8→44.4°(xmax90), η=0.599→0.526 (target 41.35°/0.5022). The earlier
xmax scans missed this because they grew xmax at FIXED nx (diluting the mesh), mixing two effects.

**Lesson:** at high scattering energy use a BALANCED box (xmax≈ymax, nx and ny grown together), never a
fixed small xmax with a large ymax. The overlap tail is not a bug. This corrects lesson #8 in the old
TODO.md "Lessons" list (the 2026-06-17 overlap-tail root cause was a misattribution of under-convergence).

**Status:** Replaced by the box-under-convergence diagnosis, then by the LL-mesh-limit finding above
(2026-06-29).
