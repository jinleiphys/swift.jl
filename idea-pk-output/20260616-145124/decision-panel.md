# Idea-PK Decision Panel — Eq.17 → swift.jl mapping (δ≈0 bug)

**Stuck-point:** Eq.17 Green-theorem n-d amplitude in swift gives δ≈0, η≈0.9, |f|×12≈4π low (converged after T4's V-protection fix).
**Run:** dual-solver (Claude contrarian + Codex methodologist), groupthink respawn fired (reduced-rep/normalization axis was undersampled), then resolved. Budget: local Julia smoke tests, weeks-not-months. Path B (direct Eq.17), magnitude must be derived not fudged.

The PK converged on a key reframing: **δ≈0 has THREE candidate causes, and they are cheaply separable in a fixed order.** Don't tune — localize first.

---

## ★ START HERE — the diagonal AND cheapest disambiguator (both = B1)

**B1 — Audit the Rxy "side move" with the explicit metric-adjoint, not the symmetric shortcut.**
The smoking gun is your own T1-vs-T4 contradiction: `bra^T·(Rxy·V·Ψ̄)` = 674 but `(Rxy·bra)^T·V·Ψ̄` = 4.4. Analytically equal only if `Rxy^T=Rxy` — which is FALSE in the non-orthogonal Laguerre metric. So at least one ordering is using the wrong adjoint.
- **Today:** in `test_eq17_green.jl:88`, replace the single `Rbra = Rxy*bra` with a printed comparison of four maps: `Rxy*bra`, `2*Rxy_13_matrix_optimized*bra`, `B\(transpose(Rxy)*B*bra)`, `B\(transpose(Rxy_31)*B*bra)` (solve `B*x=…`, don't form dense inverse). Also print `norm(B*Rxy_13 − transpose(Rxy_31)*B)/norm(…)` and the resulting `Mp` for each. nx=12,ny=30, minutes.
- **Info gain (both branches):** if a metric-correct adjoint map changes Mp's phase/scale qualitatively → the bug was moving Rxy across the metric as if symmetric (validate winner with B2). If all four leave Mp near-real and small → operator side is NOT the cause, jump to channel-projection (B5) / Born-localization (B4).
- **Risk:** B-inverse ill-conditioned on large meshes — judge by action-on-vector at small grid only.

---

## Convergent paths (both solvers proposed — high-confidence diagnostics)

**B2 — Bound-state identity as the operator ruler** *(= A3; B2 extends your existing test)*.
Before trusting `[V₂₃+V₃₁]` in Eq.17, verify it on the KNOWN identity `⟨Ψ|V₂₃+V₃₁|Ψ⟩=2⟨V₁₂⟩` on the 3-body bound state. Extend `test_V23V31_operator.jl`: print the one-pair `real(Ψ'·Rxy_31·V·Rxy_13·Ψ)/real(Ψ'·V·Ψ)` and the c-product/B-metric versions; pass = one-pair→1, two-pair→2 within truncation. **Run at θ=0 first, then add the bilinear c-product / B-metric print** (the current test uses Hermitian `Ψ'`, but CS amplitude needs bilinear). If this fails, Eq.17 cannot be right regardless of kernel/prefactor.

**B4 — Born/scattered decomposition (which piece carries the phase?)** *(= A2; B4 is the cleaner split)*.
At `test_eq17_green.jl:71-90` print magnitude+phase of `Mp_in = ⟨Rbra|V·(ψ_in+Rxy·ψ_in)⟩` and `Mp_sc = ⟨Rbra|V·(ψ_sc+Rxy·ψ_sc)⟩` separately, then the COLOSS split (unrotated Born + rotated scattered, template = your locked `test_2body_cs_1S0.jl:82-99`). **Tells you whether the scattering phase is computed-then-cancelled by the total rotated integral, or never enters `V·Ψ̄` at all.** (Refined: a tiny/aligned Mp_sc doesn't immediately implicate the source/GMRES — a wrong Rxy adjoint or collapsed channel projection also suppresses it.)

**B5 — Is `first(cpl)` throwing away the doublet channel-spin projection?** *(= A7)*.
You take `iel=first(cpl)` (a single (λ,J₃) block). The physical n-d doublet needs the full ³S₁+³D₁ channel-spin recoupling (`scattering.jl:423-542`, which `compute_scattering_amplitude` deliberately keeps). Print every `cpl` channel (λ,J₃,J₁₂,s₁₂,l), compute `Mp` per deuteron-compatible block, then recouple. **A missing recoupling weight changes BOTH phase and scale while every bra-type scan (T5) looks inert — because T5 varied the y-kernel AFTER the projection was already collapsed.** Check raw per-channel values before applying Wigner factors (avoid false sign flips).

---

## Single-source paths (the respawn axis: reduced-rep normalization + vehicle)

**A5 — Derive the y-integral factors by reducing Eq.17 to the LOCKED 2-body anchor, not by transcribing the paper's 3D factors.** *(diagonal of the normalization axis)*
Your 2-body `test_2body_cs_1S0.jl` (δ=63.512°) uses the SAME reduced Lagrange rep + COULCC. So every 3-body factor is already pinned there: regular F bra (NOT ĥ⁻), ONE e^{iθ} per *open* radial integral (the y-integral; the bound x-integral contributes none since φ_d decays), prefactor −(1/E_cm), B-metric absorbed in V. **Hypothesis this directly tests: the irregular-G-dominated ĥ⁻ bra is itself the bug — the y-kernel should be the regular F_λ the anchor uses, and C_n's phase power should be e^{iθ} not e^{3iθ}.** First write the 1-page reduced-rep ledger (4π, kernel form, #Jacobians, C_n phase), THEN make the smallest matching edit and run the 3-mesh loop.

**A6 — Synthetic Born-only unit-check to pin (mass, 4π, e^{iθ}-power).**
Set Ψ̄=ψ_in (scattering off), so Eq.17 must equal the analytically-known free d-n Born amplitude; read off the unique factor triple that makes Born come out right in swift's physical-Jacobi reduced rep. **Caveat (cross-check):** the independent Born reference must NOT reuse the same angular reduction/kernel assumption as the tested integral, or it's circular. Specify the analytic reference (V₂₃+V₃₁ geometry, bipolar-harmonic norm, deuteron c-norm, y-mass) BEFORE coding.

**A1 — Build the SPECTRAL (COLOSS) realization of Eq.17 inside swift.** *(the vehicle-change move)*
Your group's own COLOSS validates `f_sc=Σ dₙ²/(E−Eₙ)` with V-protected `dₙ=⟨V·bra|eigvec⟩`. Reuse swift's existing `(H,B)`, call `eigen(H,B)`, B-orthonormalize with the bilinear c-product, `dₙ=(Rxy·bra)^T·V·vₙ`. **If it works → δ≈0 was because the resolvent (E−H)⁻¹ was never applied (you were computing Born-like quadrature); if it fails → not a pure resolvent issue.** (Downgraded: failure doesn't isolate the bra alone — could be spectral completeness / B-orthonormalization.) Needs the derivation that this spectral form = Eq.17 in swift's basis before it's a production path. eigen(H,B) is seconds at smoke size.

**B3 — Deterministic prefactor ledger (LAST, gated).** Only after B1/B2/B5 pass: print the factor table (4π, 2μ_y/m, Jacobian n∈{1,2,3,6}, 3D vs reduced C_n) with **every factor derived from the reduced measure first — no free scans** (else it's T3 again). Separates a pure magnitude defect (~4π) from the phase/operator defect; it CANNOT explain δ≈0 if Mp stays near-real.

---

## Author note — the real fork (what to believe)

There are **three competing stories for δ≈0**, and the panel is a decision tree, not a menu:

1. **Operator-mapping bug** (B1→B2): the 674-vs-4.4 contradiction says the Rxy adjoint/metric is wrong. *Most likely given your own data.*
2. **Collapsed channel projection** (B5): `first(cpl)` drops ³D₁ → phase cancels. *Cheapest to rule in/out, and explains why every bra-scan was inert.*
3. **Wrong y-kernel/normalization** (A5/A6): irregular-G ĥ⁻ bra is wrong; the anchor says regular F. *Explains the ×4π magnitude too.*

**Recommended sequence (each step is minutes):** B1 → if inert, B5 → B4 → then A5 (kernel) → A6/B3 (magnitude). B1 and B5 together rule out two of the three stories in one afternoon. The vehicle-change A1 is the fallback if direct Eq.17 stays broken after B1+B5+A5.

**Incompatible/sequencing:** A1 vs A5 (different Eq.17 vehicles — pick one only after B1/B5). A6 before B3 (derive the Born prefactor before any factor table). A4 (Kohn integral relations) was dropped — off path-B, but it's a logged n-d-validated alternative if you ever abandon path B.
