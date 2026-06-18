# test_3body_greens.jl — 3-body multichannel n-d elastic amplitude by Lazauskas's GREEN'S-THEOREM
# method (HDR arXiv:1904.04675 Eq.2.47-2.51 / 2.117-2.118). NOT K-matrix (that route was abandoned).
#   Solve A·ψ_sc = V·Rxy·Ω_in  with  A = E·B − T − V·(I+Rxy)  (CS via back-rotation, angle θ).
#   Amplitude (Eq.2.118, Born separated):
#     f = −(1/E_cm)[ e^{2iθ}·⟨Ω_in|V·Rxy|ψ_sc⟩_CS  +  ⟨Ω_in|V·Rxy|Ω_in⟩_{Born, real axis θ=0} ]
#   bra = REGULAR incoming Ω_in = φ_d(x)F_λ(qy) (bilinear c-product, transpose, NO conjugation);
#   V·Rxy realizes the inter-cluster potential V_β+V_γ = [V₂₃+V₃₁] (Eq.2.48). S = 1 + 2i·q·f.
#   Target (Lazauskas-Carbonell PRC 84,034002 Tab.III doublet n-d, E_lab=14.1): δ=105.49°, η=0.4649.
using LinearAlgebra, Printf
const BB="/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/general_modules/channels.jl"); using .channels
include(BB*"/general_modules/mesh.jl");     using .mesh
include(BB*"/swift/matrices_optimized.jl"); using .matrices_optimized
include(BB*"/swift/scattering.jl");          using .Scattering
include(BB*"/swift/twobody.jl");             using .twobodybound
include(BB*"/swift/coulcc.jl");              using .CoulCC
const ħ=197.3269718; const m=1.0079713395678829; const amu=931.49432
fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
nθ=12; alpha=1.0; θ_deg=10.0; θ=θ_deg*π/180
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0; μ_y=2m/3

# light=true: only V and Rxy are built (Born/projection term needs no A,B,T,LU). Rxy is
# θ-independent, so the real-axis s0 reuses s's Rxy via Rxy_share instead of rebuilding the
# whole scattering matrix. This removes the entire second compute_scattering_matrix + LU.
function build(nx,ny,xmax,ymax,b; θdeg=θ_deg, lmx=2, λmx=2, j2bmx=1.0, Rxy_share=nothing, light=false)
    θl=θdeg*π/180
    α=α3b(fermion,Jtot,T,Parity,lmx,0,λmx,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,j2bmx)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    bE,bψ=bound2b(grid,potname,θ_deg=θdeg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    Efull=E_cm+E_d
    if light
        B=nothing; Tm=nothing; Tx_ch=nothing; Ty_ch=nothing; Nx=nothing; Ny=nothing; V_x_full=nothing
        V = θdeg==0.0 ? V_matrix_optimized(α,grid,potname) :
                        V_matrix_optimized_scaled(α,grid,potname,θ_deg=θdeg)
        Rxy = Rxy_share !== nothing ? Rxy_share : Rxy_matrix_optimized(α,grid;real_output=true)[1]
    else
        # assemble_A=false: no dense A / no dense V*Rxy; the CS solve goes through matrix-free GMRES
        _,B,Tm,V,Rxy,Rxy_31,Tx_ch,Ty_ch,_,Nx,Ny,V_x_full = Scattering.compute_scattering_matrix(Efull,α,grid,potname,θ_deg=θdeg,assemble_A=false)
    end
    q=sqrt(2*μ_y*amu*E_cm)/ħ
    nxg=grid.nx; nyg=grid.ny; nch=length(α.l)
    # deuteron-coupled channels grouped by λ (=l_y); j_y=J3
    chans=Dict{Int,Vector{Int}}()  # λ => [iα...]
    mcomp=Dict{Int,Int}()          # iα => deuteron comp (1=S,2=D)
    for iα in 1:nch
        i2b=α.α2bindex[iα]
        if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
            l=α.α2b.l[i2b]; mc=(l==0 ? 1 : (l==2 ? 2 : 0)); mc>0 || continue
            λ=Int(round(α.λ[iα])); push!(get!(chans,λ,Int[]),iα); mcomp[iα]=mc
        end
    end
    λs=sort(collect(keys(chans)))   # [0,2]
    # c-norm (bilinear, e^{iθ}) → pre-normalize deuteron to c-norm 1 so cħ & matrix-element terms are consistent
    n2b=size(φ_d,2); evec=ComplexF64[φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx]
    Ix=[(i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx,j in 1:grid.nx]
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θl)
    φ_d = φ_d ./ sqrt(C_n)          # c-normalized deuteron (φ_dᵀ B φ_d = 1)
    blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    # regular incoming Ω_in = φ_d(x)·F_λ(qy)/(ϕx ϕy), masked to the λ entrance group (COULCC F via helper)
    function Omega_R(λ)
        ψ=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θl)
        for iα in 1:nch; Int(round(α.λ[iα]))==λ || (ψ[(iα-1)*nxg*nyg+1:iα*nxg*nyg].=0); end
        return ψ
    end
    return (;α,grid,B,T=Tm,V,Rxy,Efull,Tx_ch,Ty_ch,Nx,Ny,V_x_full,q,C_n,λs,chans,nxg,nyg,blk,Omega_R,φd=φ_d,mcomp)
end


# ===== Rimas 2011 / HDR arXiv:1904.04675 Eq.2.117-2.118 (Green-theorem amplitude) =====
#   f = (1/Ecm)[ e^{i·p·θ} ⟨Ω_in^θ | V·Rxy | ψ_sc^θ⟩   +   ⟨Ω_in^0 | V·Rxy | Ω_in^0⟩_{NO CS} ]
# bra = REGULAR incoming Ω_in (c-product, transpose); scattered ket gets CS, BORN term (in×V×in)
# computed on the REAL axis (θ=0) since it diverges fastest with θ.  S = 1 + 2i q f → δ, η.
function run(nx,ny,xmax,ymax,b; jacpow=2.0, verbose=true, diag=false, chflux=false, lmx=2, λmx=2, j2bmx=1.0)
    s  = build(nx,ny,xmax,ymax,b; lmx=lmx,λmx=λmx,j2bmx=j2bmx)              # CS build (θ = θ_deg)
    # real-axis Born build: light (no A/B/T/LU) + share s's θ-independent Rxy
    s0 = build(nx,ny,xmax,ymax,b; θdeg=0.0,lmx=lmx,λmx=λmx,j2bmx=j2bmx, Rxy_share=s.Rxy, light=true)
    λs = s.λs
    proj(bra,ket,bld)=(acc=0.0im; for iα in bld.chans[λs[1]]; acc+=transpose(bld.blk(bra,iα))*bld.blk(ket,iα); end; acc)
    # incoming (regular F) in the doublet entrance channel
    Ω  = s.Omega_R(λs[1])                        # CS incoming
    Ω0 = s0.Omega_R(λs[1])                       # real incoming
    # scattered wave (CS): A ψ_sc = V·Rxy·Ω_in, via matrix-free preconditioned GMRES (no dense A/LU)
    RxyΩ = s.Rxy*Ω; bsrc = s.V*RxyΩ
    ψsc, _ = gmres_scattering(s.Efull, s.B, s.T, s.V, s.Rxy, bsrc, s.α, s.grid,
                              s.Tx_ch, s.Ty_ch, s.V_x_full, s.Nx, s.Ny; reltol=1e-9)
    jac = exp(im*jacpow*θ)
    f_sc  = jac*proj(Ω, s.V*(s.Rxy*ψsc), s)      # ⟨Ω|V·Rxy|ψ_sc⟩ (CS), mesh-stable for θ within constraint
    f_brn =     proj(Ω0, s0.V*(s0.Rxy*Ω0), s0)   # Born term, real axis (no CS)
    flux_norm = (μ_y/(m/2))^(1/4)
    f = -(flux_norm*cis(θ)/E_cm)*(f_sc + f_brn)
    S = 1 + 2im*s.q*f
    verbose && @printf("[lmx=%d λmx=%d j2b=%.0f nch=%2d ny=%3d ymax=%5.0f] f_sc=%.2f%+.2fi f_brn=%.2f  δ=%8.3f° η=%.4f\n",
            lmx, λmx, j2bmx, length(s.α.l), ny, ymax, real(f_sc),imag(f_sc), real(f_brn), rad2deg(0.5*angle(S)), abs(S))
    if chflux
        # source diagnostic: does V·Rxy·Ω leak OUT of the entrance sector at all?
        @printf("  per-channel source norms (Ω→Rxy·Ω→b=V·Rxy·Ω):\n")
        for iα in 1:length(s.α.l)
            @printf("    ch%2d (λ=%d): ‖Ω‖=%.3e ‖Rxy·Ω‖=%.3e ‖b‖=%.3e\n",
                iα, Int(round(s.α.λ[iα])), norm(s.blk(Ω,iα)), norm(s.blk(RxyΩ,iα)), norm(s.blk(bsrc,iα)))
        end
        # per-channel Faddeev-component flux ‖ψ_sc‖: does each channel actually couple/carry breakup?
        nrm=0.0; for iα in 1:length(s.α.l); nrm+=norm(s.blk(ψsc,iα))^2; end; nrm=sqrt(nrm)
        @printf("  per-channel ‖ψ_sc‖ (frac of total), total=%.3e:\n", nrm)
        for iα in 1:length(s.α.l)
            i2b=s.α.α2bindex[iα]
            @printf("    ch%2d (l=%d s12=%.0f J12=%.0f λ=%d J3=%.1f T12=%.0f) ‖ψ‖=%.3e (%.1f%%) ‖VRxyψ‖=%.3e\n",
                iα, s.α.α2b.l[i2b], s.α.α2b.s12[i2b], s.α.α2b.J12[i2b], s.α.λ[iα], s.α.J3[iα], s.α.α2b.T12[i2b],
                norm(s.blk(ψsc,iα)), 100*norm(s.blk(ψsc,iα))^2/nrm^2, norm(s.blk(s.V*(s.Rxy*ψsc),iα)))
        end
    end
    if diag
        # paper-derivable bra-structure variants (Eq.2.48: A=⟨Ψ_full|−(V_β+V_γ)|Ψ̃^in⟩, bi-conjugate bra).
        # All reuse Ω, ψsc; only the projection over chans[λ=0]=[2,9] (ch2=deuteron-S l2b=0, ch9=D l2b=2) changes.
        VRsc = s.V*(s.Rxy*ψsc); VRb = s.V*(s.Rxy*Ω); VRb0 = s0.V*(s0.Rxy*Ω0)
        chs  = s.chans[λs[1]]
        # generic projector: ops control conjugation, channel subset, per-channel sign
        function pj(bra,ket,bld,VRk; conj_bra=false, only=nothing, dsign=1.0)
            acc=0.0im
            for iα in chs
                (only!==nothing && iα∉only) && continue
                br = bld.blk(bra,iα); kt = VRk===nothing ? bld.blk(ket,iα) : VRk[(iα-1)*bld.nxg*bld.nyg+1:iα*bld.nxg*bld.nyg]
                sg = (bld.mcomp[iα]==2 ? dsign : 1.0)
                acc += sg*(conj_bra ? sum(conj.(br).*kt) : sum(br.*kt))
            end
            acc
        end
        report(tag,fs,fb)=(ff=-(1/E_cm)*(fs+fb); SS=1+2im*s.q*ff;
            @printf("    %-16s f_sc=%.3f%+.3fi f_brn=%.3f  δ=%8.3f° η=%.4f\n",tag,real(fs),imag(fs),real(fb),rad2deg(0.5*angle(SS)),abs(SS)))
        report("baseline",     jac*pj(Ω,nothing,s,VRsc),                 pj(Ω0,nothing,s0,VRb0))
        report("conj-bra(dot)",jac*pj(Ω,nothing,s,VRsc;conj_bra=true),   pj(Ω0,nothing,s0,VRb0;conj_bra=true))
        report("S-only(drop D)",jac*pj(Ω,nothing,s,VRsc;only=[chs[1]]),  pj(Ω0,nothing,s0,VRb0;only=[chs[1]]))
        report("D-sign-flip",  jac*pj(Ω,nothing,s,VRsc;dsign=-1.0),      pj(Ω0,nothing,s0,VRb0;dsign=-1.0))
        report("swap bra/ket", jac*pj(ψsc,nothing,s,VRb),                pj(Ω0,nothing,s0,VRb0))
        # Full wavefunction Ψ=(1+Rxy)ψ vs the single Faddeev component ψ (paper bra = full Ψ).
        ψfull = ψsc .+ s.Rxy*ψsc
        report("(1+Rxy)ψsc ket", jac*pj(Ω,nothing,s, s.V*(s.Rxy*ψfull)),  pj(Ω0,nothing,s0,VRb0))
        report("op=V only",      jac*pj(Ω,nothing,s, s.V*ψsc),            pj(Ω0,nothing,s0,(s0.V*Ω0)))
        report("op=V,(1+Rxy)ψ",  jac*pj(Ω,nothing,s, s.V*ψfull),         pj(Ω0,nothing,s0,(s0.V*Ω0)))
        @printf("    [target: f_sc+f_brn → Re≈2.05, Im≈−11.95 ⇒ δ=105.49° η=0.4649]\n")
    end
    return S
end

println("benchmark doublet 14.1: δ=105.49°, η=0.4649  [Rimas HDR Eq.2.118: scattered(CS)+Born(no CS)]")
# Derived CS Jacobian: jac=e^{2iθ} (x-contour cancels V's e^{−iθ}, y-contour supplies the missing factor).
# Check δ,η at jacpow=2, θ=3°/4°, convergence in mesh + b. Residual magnitude = deuteron c-norm / recoupling?
# θ-plateau × box-stability grid (fixed point density ny/ymax≈0.75) to locate the
# small-θ window where Im(f_sc) is box-converged. Plateau ⇒ η flat across ymax near 0.46.
# Box-stability grid at fixed point density (ny/ymax≈0.75). FINDING 2026-06-18:
#   η is box-CONVERGED at θ≥2.5° (flat across ymax 80/100/120 at ≈0.335), δ→≈104°.
#   The earlier "η drifts with ymax" was a RESOLUTION artifact (ny under-scaled), not a
#   box/CS-angle effect. So the gap to benchmark η=0.4649 is a FIXED ~28% amplitude-
#   normalization factor, NOT convergence. C_n print shows the CS deuteron c-norm carries
#   a spurious −12.4° phase vs the real-axis Born build (=1.0): the two terms in Eq.2.118
#   are differently normalized. Resolving the convention is the open question for Rimas.
# CHANNEL-TRUNCATION convergence scan (θ=3°, fixed moderate mesh). The amplitude assembly is the
# paper formula (baseline = Eq.2.48; all structural variants ruled out). η is the breakup-absorption
# observable, most sensitive to the channel space (lmx,λmx,j2bmx); δ converges faster. Question the
# framing: is η=0.334 vs benchmark 0.4649 a model-space-truncation gap, not an amplitude bug?
global θ_deg = 3.0; global θ = 3.0*π/180
# 2026-06-18: "channels are INVARIANT for MT" is CORRECT, mechanism now verified. MT is an
# S-wave-only force: ‖V_blk‖ nonzero only for l=0 channels, exactly 0 for l>0. So V·Rxy·Ω is killed
# outside the S-wave entrance — Rxy DOES couple the entrance into l>0/λ>0 channels (‖Rxy·Ω‖≠0 there)
# but V zeroes b=V·Rxy·Ω there, those channels carry zero flux, and the observable is independent of
# (lmx,λmx,j2bmax). The chflux print proves it (source norms: ‖Rxy·Ω‖≠0 but ‖b‖=0 off the l=0 sector).
# The gap to benchmark η=0.4649 is mesh + amplitude-normalization (waiting on Rimas), NOT channels.
# CAVEAT: a realistic l>0 force (AV18) WOULD couple higher channels; this MT result does not generalize.
println("=== θ=3°, per-channel source+flux diagnostic (MT S-wave-only ⇒ only l=0 entrance carries flux) ===")
println("-- nch=3 (lmx=0,λmx=2), fine mesh --")
run(24,120,30.0,120.0,8.0; lmx=0,λmx=2,j2bmx=1.0, chflux=true); flush(stdout)
# Channel-count comparisons MUST be at a FIXED mesh: comparing different meshes gives a spurious η
# shift that is a mesh artifact, not a channel effect. Same mesh ⇒ nch=2 ≡ nch=15 bit-identical for MT.
println("\n=== SAME-MESH (16,40,20,40) channel test: nch=2 vs nch=15 must match for MT ===")
println("-- nch=2 (lmx=0,λmx=0) --");   run(16,40,20.0,40.0,8.0; lmx=0,λmx=0,j2bmx=1.0); flush(stdout)
println("-- nch=15 (lmx=2,λmx=2) --");  run(16,40,20.0,40.0,8.0; lmx=2,λmx=2,j2bmx=2.0); flush(stdout)
