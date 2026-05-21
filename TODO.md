# swift.jl TODO

Snapshot of pending work as of **2026-05-21**. Picks up the session that implemented the V-sector block-diagonal M⁻¹ preconditioner.

---

## Current state (what's done)

### ✅ V-sector M⁻¹ preconditioner — implemented and cross-validated
- New code in `swift/matrices.jl`: `group_channels_by_v_sector`, `MInverseCacheVSector`, `precompute_M_inverse_cache_vsector`, `M_inverse_operator_cached_vsector`.
- New helper in `swift/matrices_optimized.jl`: `V_x_pair_blocks` (per-(i,j) channel-pair V_x extractor).
- `swift/MalflietTjon.jl`: `precompute_RHS_cache(...; use_vsector=false)`, `malfiet_tjon_solve_optimized(...; use_vsector=false)`, `compute_lambda_eigenvalue_optimized` dispatches on cache type.
- **Default still strict channel-diagonal**; pass `use_vsector=true` to opt-in.
- **Sector key**: $q = (J_{12}, T_{12}, s_{12}, \lambda_3, J_3)$ — matches `V_matrix` selection rules in `matrices.jl:235-246` and `matrices_optimized.jl:540-555` exactly. Validated by Codex audit + manual code read.

### ✅ Cross-validation against Lazauskas (rimas) — 6 jx truncations
3H AV18 (no 3NF), `lmax=8, λmax=20, nθ=12, nx=ny=20, xmax=ymax=16`:

| jx_max | N_AMP | rimas −E | swift.jl strict | swift.jl V-sector | Δ (V vs strict) |
|---|---|---|---|---|---|
| 1 | 10 | 7.196 | 7.191389 | 7.191400 | 11 μeV |
| 2 | 18 | 7.502 | 7.497110 | 7.497128 | 18 μeV |
| 3 | 26 | 7.594 | 7.588171 | — | — |
| 4 | 34 | 7.606 | 7.600041 | 7.599962 | 79 μeV |
| 5 | 42 | 7.614 | 7.608689 | — | — |
| 6 | 50 | 7.615 | 7.609668 | — | — |

Constant **5-6 keV under-binding vs rimas** at every truncation = $\hbar^2/m_N$ convention difference (rimas 41.471 vs swift.jl 41.468 MeV·fm²). Not a bug.

V-sector vs strict agreement: ≤ 100 μeV (within Arnoldi tolerance 1e-6).

V-sector Arnoldi-only speedup: **20-40%**. Total walltime speedup: 5-20% (Rxy matrix construction dominates total cost).

### ✅ Notes ↔ code consistency
`FADDEEV_R_SPACE_NOTES.md` describes V-sector M(E) preconditioner in §"Tensor-product construction of $M^{-1}$"; sector key in §"Eigenvalue problem of the Faddeev Equations" matches the code (Codex audit passed all 5 items).

### ✅ Literature wiki ingest
arXiv:1310.6631 Carbonell-Deltuva-Fonseca-Lazauskas 2013 review archived at `~/research-wiki/sources/2013-arxiv-carbonell-bound-state-techniques-multiparticle-scattering.md` with full vocab + concept-page expansion.

---

## Uncommitted state (next-time decisions)

Working tree (`git status` snapshot):

```
M  CLAUDE.md
M  NNpot/nuclear_potentials.jl     # MN potential support (added earlier session)
M  swift/MalflietTjon.jl           # V-sector RHS + dispatch
M  swift/matrices.jl               # V-sector cache + Kronecker M⁻¹
M  swift/matrices_optimized.jl     # V_x_pair_blocks
M  swift/swift_3H.jl               # j2bmax=1, strict default; comment about use_vsector
M  swift/test_scattering_amplitude.jl  # n+d (z1z2=0) fix from earlier session
?? FADDEEV_R_SPACE_NOTES.md        # renamed from "Faddeev Equations in R space..."
?? NNpot/test_mn.jl                # MN potential interface test
?? slides_presentation.md          # talk-ready slides
?? swift/swift_3H_MN.jl            # MN potential driver
?? internal_report.pdf             # (manuscript artifact — not from this session?)
?? internal_report.tex             # ditto
?? internal_reportNotes.bib        # ditto
```

**Decisions to make before commit**:
1. Bundle all V-sector + notes + MN + scattering test fixes in a single commit, or split? Suggest: split into 3 commits —
   - (a) MN potential support (NNpot changes + test_mn.jl + swift_3H_MN.jl)
   - (b) V-sector M⁻¹ preconditioner (matrices*.jl + MalflietTjon.jl + swift_3H.jl comment)
   - (c) Docs (CLAUDE.md, FADDEEV_R_SPACE_NOTES.md rename + V-sector rewrite, slides_presentation.md)
   - test_scattering_amplitude.jl fix can go with (a) or (c).
2. `internal_report.{pdf,tex,bib}` aren't from this session — figure out if they belong here or in a sibling dir.

---

## Next steps (priority order)

### High priority (immediate next session)

- [ ] **³He cross-check with V-sector mode.** Only ³H has been V-sector validated. Run `swift_3He.jl` with `use_vsector=true` at jx_max=1..6, compare against rimas's ³He table (in memory at `swift-jl-benchmark-source.md`). Expected: 5 keV under-binding from m_N convention, V-sector agreeing with strict to ~100 μeV.
- [ ] **Push 3H V-sector to jx_max=2,3,5,6** to fill in the table above (rows currently empty). Confirms the V-sector convergence plateau matches strict.
- [ ] **Decide on default mode**. If V-sector matches strict across all 12 cross-checks (³H × 6 + ³He × 6) and is consistently ≥10% faster on eigenvalue part, flip `use_vsector` default to `true` and demote strict to opt-in fallback.

### Medium priority

- [ ] **Apply V-sector to scattering solver.** Currently V-sector is only wired into `malfiet_tjon_solve_optimized` (bound-state Malfiet-Tjon). The scattering equation in `scattering.jl` uses the same M⁻¹ preconditioner via `solve_scattering_equation(...; method=:gmres)` — add `use_vsector` kwarg there too. This is the bigger win: scattering solves are more expensive than bound-state Malfiet-Tjon and benefit more from a better preconditioner.
- [ ] **UIX + V-sector compatibility check.** Current implementation puts $V_{\rm UIX}$ in RHS regardless of mode (correct because 3N force does not factor into V-sector q quantum numbers). Verify by running `swift_3H.jl` with `include_uix=true, use_vsector=true` and check ground-state energy lands at ~−8.48 MeV (experimental AV18+UIX benchmark). Currently UIX disabled in swift_3H.jl driver.
- [ ] **Profile Rxy_matrix.** At jx_max=6, Rxy dominates total walltime (22 s / 40 s). If we can cut that by 2×, total V-sector speedup becomes visible.
- [ ] **Reproduce Carbonell-Deltuva-Fonseca-Lazauskas 2013 Table 1** (n-³H at 22.1 MeV, INOY04 phase shifts and inelasticities) — this is the published cross-check anchor for the **scattering** path with complex scaling. Would also exercise V-sector mode in the scattering solver.

### Low priority / future research direction

- [ ] **Move to V-sector by default in `M_inverse_operator` (non-cached path)** for symmetry. Currently `M_inverse_operator` only does strict channel-diagonal.
- [ ] **Sparse-pattern Rxy assembly.** Rxy is dense as currently built but has known sparse structure for low partial waves; profile vs naïve dense.
- [ ] **4-body extension.** Memory layer:
      - Lazauskas-Carbonell PRC 84 (2011) + PRC 86 (2012) do 4N config-space CS Faddeev-Yakubovsky — the next natural extension of swift.jl.
      - Already discussed as plausible direction; rimas would likely cross-validate similarly.
- [ ] **GPU port** (likely via CUDA.jl or KernelAbstractions.jl). The Kronecker-product M⁻¹ application is embarrassingly parallel per sector and would benefit hugely from GPU. Lower priority until single-CPU performance is fully tuned.

---

## Key gotchas / lessons logged this session

(For details see project memory at `~/.claude/projects/-Users-jinlei-Desktop-code-swift-jl/memory/`.)

1. **Arnoldi eltype trap** (`feedback-julia-arnoldi-eltype-trap.md`): in a Krylov-friendly operator closure, allocate buffers with `Vector{eltype(v)}(undef, ...)` not `Vector{T}(undef, ...)` from the cache parametric type. ComplexF64 Arnoldi vectors fed into Float64 buffers triggers silent per-element promotion and slows the operator 100×. Cost me ~1h of debugging this session.
2. **swift.jl authorship framing** (`feedback-swift-jl-authorship.md`): describing swift.jl as "Lazauskas-Carbonell line's direct precedent" is wrong. Correct framing: "Jin's independent Julia code, following the methodological line of rimas (R-space Faddeev + Lagrange-mesh + complex scaling)".
3. **Lazauskas benchmark provenance** (`swift-jl-benchmark-source.md`): rimas provides 3H/3He AV18 (no 3NF) ground-truth via personal communication; T=1/2 approximation, $\hbar^2/m_N = 41.471$ MeV·fm². swift.jl uses 41.468 → systematic 5-6 keV under-binding at every truncation, NOT a bug.

---

*Generated at end-of-day 2026-05-21. Next session: pick up from "High priority" list above.*
