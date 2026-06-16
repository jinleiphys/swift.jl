# test_eq17_green.jl — Lazauskas-Carbonell PRC 84,034002 (2011) Eq.(17) EXACTLY, neutral n+d.
#   f_nm = -C_n^{-1}(m/ħ²) ∬ φ_n*(x e^{-iθ}) [e^{-iq y e^{iθ}}/|y|] [V_j+V_k]Ψ̄_m e^{6iθ} d³x d³y
# In swift's reduced (F/(xy)) rep:
#   - bra y-kernel e^{-iqy e^{iθ}}/|y|  ==  Riccati incoming Hankel ĥ⁻_λ = G_λ − iF_λ = gc − i·fc
#     (the /|y| is absorbed by the reduced ×y), built like compute_initial_state_vector but gc−i·fc.
#   - [V_j+V_k]Ψ̄ = Rxy·V·Ψ̄  (validated identity; V_j,V_k are the two NON-incoming pairs),
#     Ψ̄ = (1+Rxy)ψ_total, ψ_total=ψ_in+ψ_sc (full 3-body wave).
#   - C_n = ∫φ*(xe^{-iθ})φ(xe^{iθ}) e^{?θ} d³x  (reduced 1D x → e^{iθ}); MAGNITUDE is FIXED (no fudge).
# Benchmark (Tab.III doublet 14.1): Re δ=105.49°, η=0.4649  ⟹  f_target=-0.218+1.272i.
using LinearAlgebra, Printf, OffsetArrays
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
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0; μ=2m/3

BRA = :hminus   # :hminus (G−iF), :hplus (G+iF), or :Freg (regular F)
# build bra  φ_d(x)·[kernel]_λ(qy e^{iθ})/(ϕx ϕy)
function build_hankel_bra(grid,α,φ_d,q,θ)
    nx=grid.nx; ny=grid.ny; nch=length(α.l)
    bra=zeros(ComplexF64, nx*ny*nch)
    n2b=size(φ_d,2)
    # find coupling channels + λmax
    cpl=Int[]; m2b=Int[]; λmax=-1
    for iα in 1:nch
        i2b=α.α2bindex[iα]
        if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
            l=α.α2b.l[i2b]; mc=(l==0 ? 1 : (l==2 ? 2 : 0))
            if mc>0; push!(cpl,iα); push!(m2b,mc); λmax=max(λmax,Int(round(α.λ[iα]))); end
        end
    end
    sf=exp(im*θ)
    Hall=Vector{OffsetArray{ComplexF64,1}}(undef,ny)
    for iy in 1:ny
        z=ComplexF64(q*grid.yi[iy]*sf)
        fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=λmax,mode=4)
        h = if ifail!=0; zeros(ComplexF64,λmax+1)
            elseif BRA==:hminus; gc .- im.*fc      # Riccati ĥ⁻_λ = G−iF (incoming)
            elseif BRA==:hplus;  gc .+ im.*fc      # Riccati ĥ⁺_λ = G+iF (outgoing)
            else; fc end                            # regular F_λ
        Hall[iy]=OffsetArray(h,0:λmax)
    end
    for (idx,iα) in enumerate(cpl)
        λ=Int(round(α.λ[iα])); mc=m2b[idx]
        for ix in 1:nx, iy in 1:ny
            i=(iα-1)*nx*ny+(ix-1)*ny+iy
            fx=grid.ϕx[ix]; fy=grid.ϕy[iy]
            (abs(fx)<1e-15||abs(fy)<1e-15) && continue
            bra[i]=(φ_d[ix,mc]*Hall[iy][λ])/(fx*fy)
        end
    end
    return bra, cpl
end

function run_case(nx,ny,xmax,ymax)
    α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
    grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    V           = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
    Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
    bE,bψ = bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    ψ_in = compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
    ψ_sc,A,b = solve_scattering_equation(E_cm+E_d,α,grid,potname,ψ_in,θ_deg=θ_deg)
    q=sqrt(2*μ*amu*E_cm)/ħ
    # full wave Ψ̄ = (1+Rxy)ψ_total ; operator [V_j+V_k]Ψ̄ = Rxy·V·Ψ̄
    ψ_total=ψ_in.+ψ_sc; Ψbar=ψ_total.+Rxy*ψ_total; W=Rxy*(V*Ψbar)
    # ψ_sc-only variant (scattered Faddeev amp DECAYS under CS → integral should CONVERGE)
    Ψbar_sc=ψ_sc.+Rxy*ψ_sc; W_sc=Rxy*(V*Ψbar_sc)
    # C_n (reduced 1D x → e^{iθ})
    n2b=size(φ_d,2)
    evec=ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
    Ix=[ (i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)
    bra,cpl=build_hankel_bra(grid,α,φ_d,q,θ)
    nxg=grid.nx; nyg=grid.ny; blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    iel=first(cpl)
    # OLD (diverges): bra^T·(Rxy·V·Ψ̄) — V not adjacent to growing bra
    M=transpose(blk(bra,iel))*blk(W,iel)
    M_sc=transpose(blk(bra,iel))*blk(W_sc,iel)
    # CORRECT P-op algebra: ⟨bra|V₂₃+V₃₁|Ψ̄⟩ = (Rxy·bra)ᵀ·V·Ψ̄  → V multiplies the rearranged
    # bra (short-range WS auto-truncates the growing Hankel, COLOSS-style V·fc protection).
    Rbra = Rxy*bra
    VΨ   = V*Ψbar          # V·Ψ̄ confined by short-range V
    Mp   = transpose(blk(Rbra,iel))*blk(VΨ,iel)
    return (grid,q,C_n,M,M_sc,Mp,iel,blk,bra,W)
end

println("benchmark: Re δ=105.49°, η=0.4649,  f_target=-0.218+1.272i,  m/ħ²=", round(m/(ħ^2/amu),sigdigits=4))
for BRAtype in (:hminus,:hplus,:Freg)
    global BRA=BRAtype
    println("\n########## bra = $BRAtype  (V-protected ordering (Rxy·bra)ᵀ·V·Ψ̄) ##########")
    for (nx,ny,xmax,ymax) in [(12,30,24.0,50.0),(16,40,28.0,60.0),(20,50,32.0,70.0)]
        g,q,C_n,M,M_sc,Mp,iel,blk,bra,W = run_case(nx,ny,xmax,ymax)
        base_p = -(m/(ħ^2/amu))/C_n * Mp
        f0=base_p; S0=1+2im*q*f0
        @printf("nx=%2d ny=%2d : |Mp|=%.3e  |f|=%.3f  (n=0) Re δ=%8.3f° η=%.4f\n",
                nx,ny,abs(Mp),abs(f0),rad2deg(0.5*angle(S0)),abs(S0))
    end
end
