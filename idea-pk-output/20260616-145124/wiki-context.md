## Read entries (papers / methods / systems already in the user's wiki on this topic)

- [[2011-prc-lazauskas-carbonell-cs-few-body-scattering]] — THE target paper (PRC 84,034002). Wiki note records the EXACT structure + benchmark:
  - "scattered amplitudes recovered either by orthogonal-channel projection (Eq.16) or by **Green's-theorem integrals on bound × scattered overlap integrals (Eq.17 + 19)**".
  - benchmark: **n-d doublet δ(²S)=105.50° at 14.1 MeV, η=0.4653 (Table III)**; quartet δ(⁴S)=68.97°, η=0.9784. Agree w/ Witała1995 / Deltuva2005 to 3 sig figs.
  - **Eq.(18) convergence bound: tan θ < √3 k_m/(q_m+2q_n) → θ_max≈14.2° at 14.1 MeV** (user runs θ=10°, WITHIN bound — so θ=10° is legal, not the problem).
  - **Numerics: piecewise Hermite-spline, 30–40 splines/direction → 3-digit accuracy** (swift uses Laguerre, DIFFERENT basis; spline boundary handling may matter).
  - documented caveat: "**Eq.(19) converges slowly in the y direction**" (small discrepancy at breakup ϑ→90°). ⟹ slow y-convergence of the Green integral is a KNOWN feature, not necessarily a bug — but user sees DIVERGENCE (worse than slow), so still anomalous.
  - 2-body: MT ¹S₀ δ=63.512° to 5 digits at r_max=100, θ=10° (matches user's locked test_2body_cs_1S0.jl).

- [[2003-phd-lazauskas-faddeev-yakubovsky-few-body-scattering]] — Lazauskas thesis. User has it (read §2.3 scattering BC + §1.1.4 2-body integral phase shift this session). Has the full FY partial-wave amplitude/observables appendix.

- [[2019-hdr-lazauskas-complex-scaling-scattering]] — Lazauskas HDR (arXiv:1904.04675). "Smooth CS turns N-body scattering into bound-state-like L² problem, validated ~3 digits vs momentum-space AGS for n+³H above breakup." Likely has the cleanest amplitude-extraction write-up.

- [[methods/complex-scaled-greens-function]] — CSGF (Kato spectral variant). NOTE: "**Direct predecessor of the COLOSS package**." COLOSS's f_sc=Σ dn²/(E−Eₙ) IS this spectral Green's function. So COLOSS's V-protected `dn=⟨V·fc|eigvec⟩` is the spectral realization of the same Green integral.

- [[methods/integral-relations]] ([[2009-prl-integral-relations-three-body-continuum-barletta]], [[2012-pra-integral-relations-1plus2-breakup-garrido]]) — ALTERNATIVE interior-based amplitude extraction: **two Kohn-variational integral relations give tan δ as a ratio of matrix elements from the INTERIOR wave function**, converges bound-state-like, AVOIDS asymptotic extraction. Garrido2012 benchmarked exactly on **n-d breakup**. This is a DIFFERENT route from Eq.17 Green's theorem (user picked path B = Eq.17), but it's a logged, n-d-validated interior-extraction the user knows — flag if a solver proposes "use Kohn/variational integral relations" so it's surfaced as a known alternative, not reinvention.

- [[methods/faddeev-three-body]], [[2013-arxiv-carbonell-bound-state-techniques-multiparticle-scattering]] (the review, refs [41,42,44,50,51] = the line swift.jl follows).

## Open contradictions / questions (from the wiki itself)

- No logged debate directly on CS amplitude extraction (debates/ only has post-prior-controversy, unrelated).
- Implicit tension surfaced by the wiki note: the paper claims Eq.17 Green-theorem "integrand vanishes outside the interaction region → accuracy improved ~1 digit over asymptotic", yet the SAME paper flags Eq.(19) "converges slowly in the y direction". So even the authors' own Green integral has a y-convergence subtlety — the user's DIVERGENCE (vs their slow-but-converging) points to a discretization/operator-mapping error specific to swift's Laguerre basis, consistent with the user's own diagnosis (Rxy ordering / truncated-basis non-symmetry).
- COLOSS = spectral realization (Σdn²/(E−Eₙ)) of this Green integral with V-protected bra; the user's direct-quadrature realization of Eq.17 is a different numerical route to the same physics — the spectral route is the one the user's group has actually validated.
