# Idea-PK Stuck Card

## Goal

User (Jin Lei, Tongji University full professor, nuclear-theory PI; SCNT IMP-CAS Huizhou visiting scientist 2026-05 to 2027-10) wants to identify **research scenarios for a 12-month single-shot PRL on three-body / few-body nuclear physics** building on the existing capabilities of his Julia code `swift.jl` (R-space Faddeev + Lagrange-mesh + complex scaling + COULCC, Lazauskas-Carbonell methodological line).

Constraint: **PRL tier (4-page Letter, single punchline, broad-significance bar), 12 months**, must clear PRL "broad nuclear-physics audience" threshold, not just PRC technical novelty.

## Current state (capabilities + ecosystem the user already owns)

**swift.jl operational capabilities (verified 2026-05):**
- Bound state of ³H/³He with realistic NN (AV18/AV14/Nijmegen) + UIX three-nucleon force; direct generalized-eigenvalue + Malfliet-Tjon iteration both work.
- ³H/³He bound state with Minnesota (MN) potential u=1.0 Serber; benchmarked against NCM VMC reference E(³H) = -7.889 ± 0.047 MeV.
- n+d elastic scattering above breakup threshold with Malfliet-Tjon (MT) potential at E = 1 MeV using complex scaling (θ_deg = 8°); LU + preconditioned GMRES paths both work.
- Partial-wave scattering amplitude matrix + Blatt-Biedenharn eigenphase shifts + mixing parameters (ε, ζ, η) on channel-spin basis.
- COULCC Fortran wrapper for Coulomb wavefunctions F_λ, G_λ (charged channels ready, currently n+d test path is z1z2=0).
- V-sector M⁻¹ block-diagonal preconditioner (Lazauskas split) committed 2026-05.
- Public GitHub repo at github.com/jinleiphys/swift.jl.

**Author's other 3-body / few-body relevant capabilities + code lines:**
- **Line A (IAV / CDCC inclusive breakup, daily driver)**: SMOOTHIE Fortran 95 code (github.com/jinleiphys/smoothie), 14-paper IAV-DWBA + IAV-CDCC research line since 2015. Two students (Liu Hao, Liu Junzhe) actively running.
- **Line B (Threshold anomaly / phenomenology)**: frozen since 2019.
- **Line C (Few-body universality / Faddeev-AGS in P-space)**: PRC 98, 051001(R) 2018 "6Li is a deuteron halo"; uses P-space AGS + EST separable expansion (Hlophe-Elster-Lei line). Has been frozen since 2019.
- **Line D (Statistical inference / UQ)**: 1 paper (Navarro Pérez & Lei 2019 PLB 795).
- **Line E (Bound-state techniques + emulators, ACTIVE)**: 4 papers including PRC 113, 024614 (DBMM 2026 Lei sole) + PRC 113, 044610 (reduced-basis emulator 2026 Lei sole). SLAM.jl Julia DBMM code. Mature emulator infrastructure on top of DBMM.
- **Line F (Coupled-channel absorption mechanism, EMERGING)**: Liu Hao + Lei + Ren 2026 PRC 113 054601 + PLB 877 140479; arXiv:2508.07584 (Aug 2025) absorption-trilogy completion. Generalized optical theorem + IWBC + complex-W decomposition.
- **inhomoR**: Lagrange-mesh R-matrix Fortran (github.com/jinleiphys/inhomoR), Brussels-Pisa code line.
- **HPC**: heliumx / alpha / helium / BSCC clusters; serious compute available.
- **Students**: 2 PhD (Liu Hao 3 PRC + 1 PLB + 1 preprint, Liu Junzhe 1 PRC + COLOSS code 2025) + 1 MS (Lu Yazhou 1 PRC).
- **Collaborators**: Moro (Sevilla, PhD advisor), Elster (Ohio, postdoc advisor), Bonaccorso (Pisa, retired 2026), Descouvemont (Brussels), Ren Z.Z. (Tongji group leader), Phillips/Nunes/Hlophe/Nogga (Line C alumni).

## Tried (constructed from research-profile + literature-wiki as the user's de facto already-done list)

### raw
User did not separately list. Reconstructed from profile + CLAUDE.md + literature-wiki.

### normalized

- **T1.** P-space Faddeev-AGS + EST separable expansion for 6Li-as-d+α universality (Lei et al. PRC 98, 051001(R) 2018 sole-first; PRC 96 + PRC 100 with Hlophe; Phillips-line analog in (a_dα, E_6Li)). [status: succeeded-but-insufficient — published flagship, but research line frozen since 2019 with no follow-up. Reopening requires new physics.]
- **T2.** R-space Faddeev for ³H/³He bound state with AV18 / AV14 / Nijmegen / MN + UIX. swift.jl operational. [status: succeeded; benchmarked to 0.2 keV vs reference values.]
- **T3.** R-space Faddeev for n+d elastic scattering above breakup threshold with MT potential + complex scaling (θ_deg = 8°). swift.jl operational. **NOTE**: this currently uses MT (Malfliet-Tjon toy potential), NOT realistic NN. Realistic-NN n+d above-breakup is not yet demonstrated in swift.jl. [status: partial — algorithmic infrastructure done, realistic-NN application pending.]
- **T4.** IAV inclusive breakup as effective 3-body Faddeev DWBA / IAV-CDCC (Line A spine). 14 papers since 2015. [status: succeeded — daily-driver research line, but Line A's PRLs (2019 PRL 122 / PRL 123) were on IAV mechanism, not strict 3N benchmarks.]
- **T5.** 6Li Phillips-line universality (a_dα, E_6Li) flagship 2018 PRC. [status: succeeded-but-insufficient.]
- **T6.** Trojan Horse Method analytical bridge to IAV (Lei 2026 sole 7th paper, submission-ready draft). [status: in-flight; spectral-framework PRC + companion PRL on ¹⁹F(p,αγ)¹⁶O 11 keV resonance planned.]
- **T7.** DBMM bound-state technique + reduced-basis emulator for elastic CDCC (Lei 2026 sole PRC 113 024614 + 044610). [status: succeeded — Line E foundation.]
- **T8.** Liu Hao + Lei + Ren absorption mechanism trilogy (Line F): generalized optical theorem decomposition, IWBC + complex-W identity. [status: succeeded.]
- **T9.** "Spectroscopic quenching of dynamical origin" (Lei 2026 sole PRL LP19345, under revision after referee report). [status: under review, not on the publish-this-year list since already in pipeline.]
- **T10.** CDCC channel-importance via DPP decomposition (Lei + Liu Hao 2026 PRC Letter CRR1074, round-1 referee report received 2026-05-20, revision drafted). [status: in-flight.]
- **T11.** Profile explicitly names as "natural next step" the **4N Yakubovsky / Borromean halo CS / CDCC-3body interface** for swift.jl. Not yet attempted.

### Already-read literature (Phase 0.5 wiki synthesis surfaced; do NOT propose paths that just reproduce these)

- **Lazauskas & Carbonell 2011 PRC 84, 034002** — CS Faddeev + Faddeev-Merkuriev on n-d / p-d with MT I-III; 3-digit benchmark vs Witała 1995 / Deltuva 2005 PRC 71. **swift.jl is reproducing this paper as its v0 milestone.**
- **Lazauskas-Carbonell 2012 PRC 86, 044002** — CS Faddeev-Yakubovsky on n-³H above 4N breakup, but only MT I-III (not realistic NN).
- **Deltuva-Fonseca 2007/2012/2013** — P-space AGS + complex-energy + spline-special-weights for fully realistic 4N scattering above breakup. **The only existing realistic-NN above-4N-breakup tool.** Carbonell 2013 review explicitly names this.
- **Carbonell-Deltuva-Fonseca-Lazauskas 2013 arXiv:1310.6631** — 36-pp 6-method taxonomy review.
- **Zhang & Furnstahl 2022 PRC 105, 064004** — EC + KVP for fast 3-body scattering emulator (1-parameter demonstration; subscript "Bayesian-ready" not yet realized).
- **König et al. 2020 PLB 810, 135814** — first EC emulator for multi-parameter chiral-EFT ab initio NCSM bound states (³H, ⁴He). NOT scattering.
- **Viviani-Kievsky-Rosati 2005 PRC 71, 024006** — precision HH + KVP for ⁴He with realistic NN+3N. Pisa line.
- **Wiringa-Arriaga-Pandharipande 2003 PRC 68, 054006** — AV18 vs AV18pq same phase shifts, different short-range wave functions (3-body / 4-body wave-function dependence open question).
- **Carlson et al. 2015 RMP 87** — QMC nuclear physics review (GFMC / AFDMC benchmark).
- **NuPECC 2024 LRP** — names "4b-IAV", "4-body CDCC for Borromean projectiles", IAV explicit, ML/AI/QC chapter.
- **NSAC 2023 LRP** — Bayesian + AI/ML + emulator as the §4.6 Line E charter sentence.
- **Kato-Aoyama-Myo-Ikeda CSGF line** — 3-body model of ⁶He / ¹¹Li Coulomb dissociation E1 strength (simple α+n+n + KKNN + Minnesota).

### Open questions explicitly flagged in the read corpus that are PRL candidate territory

1. **KVP-on-bound-state-basis validity above 3N breakup threshold** — Carbonell 2013 §3 closing: "Method validity above 3N breakup not yet established".
2. **Whether complex-energy method in configuration space can compete with momentum-space at realistic-NN level** — Carbonell 2013 §8 (config-CE only demonstrated for e+H).
3. **realistic-NN R-space 4N above-breakup Yakubovsky** — Carbonell 2013 explicitly: "the only existing technique for fully-realistic 4N scattering above the 4N breakup threshold" is Deltuva-Fonseca P-space; R-space realistic-NN side is empty.
4. **Adaptation of bound-state-code optimal variational parameters to scattering's incoming-wave structure** — Carbonell 2013 §8: "under-developed integration step".
5. **AV18 vs AV18pq wave-function disambiguating observable** — Wiringa 2003: same phase shifts, different short-range wave functions; what observable tells them apart?
6. **6He / 11Li full 4-body Coulomb dissociation with realistic NN + 3NF + EM operators** — Kato CSGF currently only 3-body with simple potentials.
7. **Multi-parameter chiral-3NF-LEC scattering emulator at A=3 with full Bayesian UQ** — König 2020 bound state only; Zhang-Furnstahl 2022 1-parameter only.
8. **4b-IAV / 4-body CDCC for Borromean projectiles** — NuPECC 2024 Box 4.4 priority; ties to Jin's Line A explicitly.

## Budget / constraints

- **Time**: 12 months from 2026-05-21 to ~2027-05.
- **Journal**: Physical Review Letters (4-page Letter, single-punchline format, broad-significance bar, ~10 figs max, no software-package names in body, single-author papers use "I").
- **Compute**: heliumx / alpha / helium / BSCC HPC clusters available; GPU + multi-node OK; cost-aware ("no 空跑").
- **Code budget**: swift.jl Julia 3-body branch active; inhomoR Fortran R-matrix; SLAM.jl DBMM Julia; STARS Fortran/CUDA. Three Tongji bound-state-techniques codes (Lei + students).
- **Manpower**: Lei + 2 PhD + 1 MS + collaborators (Moro / Ren / Descouvemont / Phillips / Nunes). Lei is also Visiting Scientist at SCNT IMP-CAS Huizhou 2026-05 to 2027-10 (extra affiliation for paper publishing during this window).
- **Hard physics constraints**:
  - No em-dashes (Chinese 无 ——, English no `—`).
  - "Physics correctness > everything"; "Run-then-think"; "anti-overengineering"; "papers are arguments not technical reports"; "cross-AI validation default".
  - PRL must clear "broad nuclear-physics audience appeal" bar, not just methodological novelty.
- **No-go zones**:
  - Path that requires building a new code line from scratch (>3 months on infrastructure) is too expensive.
  - Path that depends on unpublished experimental data from collaborators Lei does not own.
  - Path that is purely "reproduce Lazauskas-Carbonell 2011 with AV18 in n+d above breakup" — this is the swift.jl v0 milestone, not a PRL.
  - Path that re-runs Kato-Aoyama-Myo CSGF 3-body model of ⁶He E1 — already done in the read corpus.
