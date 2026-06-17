# test_3body_kmatrix.jl — 3-body multichannel n-d K-matrix (Duerinck §1.3.5, Eq.1.342-1.357).
# Method (validated 2-body analog test_2body_kmatrix.jl): swift A=E·B−T−V·(I+Rxy) == [EN−H0−Vx−P].
#   (R): A·v_c^(R) = V·Rxy·Ω_c^(R)              [Ω_c^(R)=φ_d(x)·F_{l_y}(qy), masked to λ=l_y]
#   (I): A·v_c^(I) = B·d3_c + V·Rxy·Ω̃_c^(I)      [Ω̃ uses G̃_{l_y}=G·reg; d3=φ_d⊗d_y localized defect]
#        d_y = −E_cm[2 G_{l_y}'(z) reg_z' + G_{l_y}(z) reg_z''], z=qy e^{iθ}, reg_z=1−e^{−z/(qb)}
#   U^(λ)_c = Ω_c^(λ) + v_c^(λ).  K from [ (ℏ²q/2µ_y)δ + M^(I) ] K = −M^(R),
#   M^(λ)_{c,c'} = (jac/C_n)·Σ_{iα∈c} ⟨Ω_c^(R)(iα)| V·Rxy·U^(λ)_{c'} (iα)⟩  (bilinear).
#   S=(I+iK)(I−iK)^{-1}, diagonalize → doublet (λ=0-dominant) eigenphase. Target δ=105.49°, η=0.4649.
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
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0; μ_y=2m/3
ħ2_2μy = ħ^2/(2*μ_y*amu)   # = 31.10 MeV·fm²
ħ2_over_m = ħ^2/(m*amu)    # = 41.471 MeV·fm² (ℏ²/m0, m0 = nucleon mass; RHS of Eq.1.357)

function build(nx,ny,xmax,ymax,b)
    α=α3b(fermion,Jtot,T,Parity,2,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    bE,bψ=bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    A,B,Tm,V,Rxy,Rxy_31,Tx_ch,Ty_ch,_,Nx,Ny,_ = Scattering.compute_scattering_matrix(E_cm+E_d,α,grid,potname,θ_deg=θ_deg)
    q=sqrt(2*μ_y*amu*E_cm)/ħ
    nxg=grid.nx; nyg=grid.ny; nch=length(α.l)
    # y-kinetic full matrix (block-diagonal in channel): H0y = block-diag kron(Nx, Ty_ch[iα]) (CS-scaled e^{−2iθ})
    Nfull=nxg*nyg*nch; Ty_full=zeros(ComplexF64,Nfull,Nfull)
    for iα in 1:nch
        rng=(iα-1)*nxg*nyg+1:iα*nxg*nyg; Ty_full[rng,rng]=kron(ComplexF64.(Nx),ComplexF64.(Ty_ch[iα]))
    end
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
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)
    φ_d = φ_d ./ sqrt(C_n)          # c-normalized deuteron (φ_dᵀ B φ_d = 1)
    # COULCC G_{l_y}, G' on y-mesh (mode=1)
    λmax=maximum(λs)
    Gy=Vector{OffsetArray{ComplexF64,1}}(undef,nyg); Gyp=similar(Gy)
    for iy in 1:nyg
        z=ComplexF64(q*grid.yi[iy]*exp(im*θ)); fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=λmax,mode=1)
        Gy[iy]=OffsetArray(ifail==0 ? gc : zeros(ComplexF64,λmax+1),0:λmax)
        Gyp[iy]=OffsetArray(ifail==0 ? gcp : zeros(ComplexF64,λmax+1),0:λmax)
    end
    qb=q*b
    blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    # builders for channel c (=λ): Ω^(R) (regular F via compute_initial_state_vector masked),
    # Ω̃^(I) (deuteron×G̃), d3 (deuteron×d_y)
    function Omega_R(λ)
        ψ=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
        for iα in 1:nch; Int(round(α.λ[iα]))==λ || (ψ[(iα-1)*nxg*nyg+1:iα*nxg*nyg].=0); end
        return ψ
    end
    function Omega_I_and_d3(λ)
        Ω=zeros(ComplexF64,nxg*nyg*nch); d3=zeros(ComplexF64,nxg*nyg*nch)
        for iα in chans[λ]
            mc=mcomp[iα]
            for ix in 1:nxg, iy in 1:nyg
                i=(iα-1)*nxg*nyg+(ix-1)*nyg+iy; fx=grid.ϕx[ix]; fy=grid.ϕy[iy]
                (abs(fx)<1e-15||abs(fy)<1e-15) && continue
                z=ComplexF64(q*grid.yi[iy]*exp(im*θ)); e=exp(-z/qb); om=1-e
                p=2*λ+1   # regularization power (1−e^{−r/b})^{2l+1}
                regz=om^p; regzp=p*om^(p-1)*(e/qb); regzpp=(p/qb^2)*((p-1)*om^(p-2)*e^2 - om^(p-1)*e)
                Gt=Gy[iy][λ]*regz
                dy=-E_cm*(2*Gyp[iy][λ]*regzp + Gy[iy][λ]*regzpp)
                Ω[i]=(φ_d[ix,mc]*Gt)/(fx*fy)
                d3[i]=(φ_d[ix,mc]*dy)/(fx*fy)
            end
        end
        return Ω,d3
    end
    Nxc=ComplexF64.([(i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:nxg,j in 1:nxg])
    return (;α,grid,A,B,V,Rxy,q,C_n,λs,chans,nxg,nyg,blk,Omega_R,Omega_I_and_d3,Ty_full,Erel=E_cm,φd=φ_d,mcomp,Nx=Nxc)
end

function run(nx,ny,xmax,ymax,b; jacpow=1.0, useCn=true, diag=false)
    s=build(nx,ny,xmax,ymax,b); λs=s.λs; nc=length(λs)
    VRxy(u)=s.V*(s.Rxy*u)
    ΩR=Dict(λ=>s.Omega_R(λ) for λ in λs)
    vR=Dict(); vI=Dict(); d3d=Dict(); ΩtI=Dict()  # interior ϕ^(R), ϕ^(I), defect, and Ω̃^I (deuteron×G̃)
    for λ in λs
        vR[λ]=s.A\VRxy(ΩR[λ])
        Ω̃,d3=s.Omega_I_and_d3(λ); d3d[λ]=d3; ΩtI[λ]=Ω̃; vI[λ]=s.A\(s.B*d3 .+ VRxy(Ω̃))
    end
    if diag
        # deuteron x-PROJECTION of the elastic channel, then ratio to outgoing h⁺=G̃+iF.
        # u(y)=Σ_iα φ_d^T·Nx·v_block[:,iy] ; project v^R, Ω^R(=F), Ω̃^I(=G̃) identically (norm cancels in ratio).
        function projy(v)            # → length-ny elastic-channel profile (bilinear deuteron overlap in x)
            u=zeros(ComplexF64,s.nyg)
            for iα in s.chans[λs[1]]
                M=reshape(s.blk(v,iα),s.nyg,s.nxg)          # M[iy,ix]
                px=transpose(s.φd[:,s.mcomp[iα]])*s.Nx       # 1×nx
                u .+= M*vec(px)
            end
            return u
        end
        uv=projy(vR[λs[1]]); uF=projy(ΩR[λs[1]]); uG=projy(ΩtI[λs[1]])
        hp=uG.+im.*uF
        println("  y-scan:  iy   y      |u_v|       |h⁺|      |u_v|/|h⁺|   arg(u_v/h⁺)")
        for iy in round.(Int, range(div(s.nyg,3), s.nyg, length=10))
            iy=min(iy,s.nyg); r=uv[iy]/hp[iy]
            @printf("        %3d  %5.1f  %.3e  %.3e   %.4e   %+.1f°\n",
                    iy, s.grid.yi[iy], abs(uv[iy]), abs(hp[iy]), abs(r), rad2deg(angle(r)))
        end
    end
    # ===== Rimas 2011 Eq.17 (Green theorem) — the method Lazauskas actually uses =====
    #   f_nm = −C_n⁻¹ (2µ_y/ħ²) e^{6iθ} ∫∫ φ_n*(xe^{−iθ}) [exp(−iq y e^{iθ})/|y|] [V_j+V_k] ψ_m d³x d³y
    # ket ψ_m = full Faddeev amplitude u^R (= Ω^R + v^R); operator [V_j+V_k] = V·Rxy (localizes (x,y)).
    # bra = φ_d × OUTGOING exp(−iq y e^{iθ}) (NOT regular F); c-product (transpose) + deuteron already c-normed.
    # S = 1 + 2i q f → δ,η.  [V_j+V_k] localizes → should be MESH-STABLE (unlike K-matrix volume integral).
    # outgoing bra: φ_d(x)·exp(−iq y e^{iθ}) / (ϕx ϕy), on λ=0 deuteron channels
    bra=zeros(ComplexF64, s.nxg*s.nyg*length(s.α.l))
    for iα in s.chans[λs[1]]
        mc=s.mcomp[iα]
        for ix in 1:s.nxg, iy in 1:s.nyg
            i=(iα-1)*s.nxg*s.nyg+(ix-1)*s.nyg+iy; fx=s.grid.ϕx[ix]; fy=s.grid.ϕy[iy]
            (abs(fx)<1e-15||abs(fy)<1e-15) && continue
            bra[i]=(s.φd[ix,mc]*exp(-im*s.q*s.grid.yi[iy]*exp(im*θ)))/(fx*fy)
        end
    end
    # WHICH CHANNEL carries the non-decaying source plateau? per-channel ‖src(iα,y)‖ at small vs large y.
    Ω=ΩR[λs[1]]; src=VRxy(Ω); RΩ=s.Rxy*Ω
    nch=length(s.α.l)
    chn(w,iα,iy)=sqrt(sum(abs2(w[(iα-1)*s.nxg*s.nyg+(ix-1)*s.nyg+iy]) for ix in 1:s.nxg))
    iy_s=div(s.nyg,3); iy_L=s.nyg-1
    @printf("    y_small=%.1f  y_large=%.1f   (decay ratio = small→large)\n", s.grid.yi[iy_s], s.grid.yi[iy_L])
    println("    iα  λ  l2b   ‖src(y_s)‖  ‖src(y_L)‖   ratio    ‖Rxy·ΩR(y_L)‖")
    for iα in 1:nch
        λi=Int(round(s.α.λ[iα])); l2b=s.α.α2b.l[s.α.α2bindex[iα]]
        ss=chn(src,iα,iy_s); sL=chn(src,iα,iy_L); rΩL=chn(RΩ,iα,iy_L)
        (ss<1e-12 && sL<1e-12) && continue
        @printf("   %3d  %d   %d    %.3e  %.3e   %.2e   %.3e\n", iα,λi,l2b, ss, sL, sL/max(ss,1e-30), rΩL)
    end
end
println("benchmark doublet 14.1: δ=105.49°, η=0.4649  [Rimas 2011 Eq.17 Green-theorem amplitude]")
# 2026-06-17: switched from the (wrong, overcomplicated) Duerinck-thesis K-matrix to Rimas's actual
# Eq.17. Key fixes: operator [V_j+V_k]=V·Rxy (localizes (x,y) → mesh-stable), bra = φ_d×outgoing
# exp(−iq y e^{iθ}) (NOT regular F), ket = full ψ_total. KEY TEST: is M17 mesh-stable? then calibrate jac.
for (nx,ny,xmax,ymax,b) in [(20,80,30.0,120.0,8.0)]
    @printf("[nx=%d ny=%d ymax=%.0f]\n", nx,ny,ymax); run(nx,ny,xmax,ymax,b)
end
