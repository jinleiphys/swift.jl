# swift.jl devlog ARCHIVE (older dead-ends, NOT auto-loaded)

Reverse-chronological. Moved out of devlog.md to keep the auto-loaded portion small.

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
