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

function build(nx,ny,xmax,ymax,b; θdeg=θ_deg)
    θl=θdeg*π/180
    α=α3b(fermion,Jtot,T,Parity,2,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    bE,bψ=bound2b(grid,potname,θ_deg=θdeg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    A,B,Tm,V,Rxy,Rxy_31,Tx_ch,Ty_ch,_,Nx,Ny,_ = Scattering.compute_scattering_matrix(E_cm+E_d,α,grid,potname,θ_deg=θdeg)
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
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θl)
    φ_d = φ_d ./ sqrt(C_n)          # c-normalized deuteron (φ_dᵀ B φ_d = 1)
    # COULCC G_{l_y}, G' on y-mesh (mode=1)
    λmax=maximum(λs)
    Gy=Vector{OffsetArray{ComplexF64,1}}(undef,nyg); Gyp=similar(Gy)
    for iy in 1:nyg
        z=ComplexF64(q*grid.yi[iy]*exp(im*θl)); fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=λmax,mode=1)
        Gy[iy]=OffsetArray(ifail==0 ? gc : zeros(ComplexF64,λmax+1),0:λmax)
        Gyp[iy]=OffsetArray(ifail==0 ? gcp : zeros(ComplexF64,λmax+1),0:λmax)
    end
    qb=q*b
    blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    # builders for channel c (=λ): Ω^(R) (regular F via compute_initial_state_vector masked),
    # Ω̃^(I) (deuteron×G̃), d3 (deuteron×d_y)
    function Omega_R(λ)
        ψ=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θl)
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
                z=ComplexF64(q*grid.yi[iy]*exp(im*θl)); e=exp(-z/qb); om=1-e
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


# ===== Rimas 2011 / HDR arXiv:1904.04675 Eq.2.117-2.118 (Green-theorem amplitude) =====
#   f = (1/Ecm)[ e^{i·p·θ} ⟨Ω_in^θ | V·Rxy | ψ_sc^θ⟩   +   ⟨Ω_in^0 | V·Rxy | Ω_in^0⟩_{NO CS} ]
# bra = REGULAR incoming Ω_in (c-product, transpose); scattered ket gets CS, BORN term (in×V×in)
# computed on the REAL axis (θ=0) since it diverges fastest with θ.  S = 1 + 2i q f → δ, η.
function run(nx,ny,xmax,ymax,b; jacpow=2.0, verbose=true)
    s  = build(nx,ny,xmax,ymax,b)               # CS build (θ = θ_deg)
    s0 = build(nx,ny,xmax,ymax,b; θdeg=0.0)      # real-axis build for the Born term
    λs = s.λs
    proj(bra,ket,bld)=(acc=0.0im; for iα in bld.chans[λs[1]]; acc+=transpose(bld.blk(bra,iα))*bld.blk(ket,iα); end; acc)
    # incoming (regular F) in the doublet entrance channel
    Ω  = s.Omega_R(λs[1])                        # CS incoming
    Ω0 = s0.Omega_R(λs[1])                       # real incoming
    ψsc = s.A \ (s.V*(s.Rxy*Ω))                  # scattered wave (CS): A ψ_sc = V·Rxy·Ω_in
    jac = exp(im*jacpow*θ)
    f_sc  = jac*proj(Ω, s.V*(s.Rxy*ψsc), s)      # ⟨Ω|V·Rxy|ψ_sc⟩ (CS), mesh-stable for θ within constraint
    f_brn =     proj(Ω0, s0.V*(s0.Rxy*Ω0), s0)   # Born term, real axis (no CS)
    f = -(1.0/E_cm)*(f_sc + f_brn)
    S = 1 + 2im*s.q*f
    verbose && @printf("[ny=%3d ymax=%5.0f] jac e^{%.0fθ}: f_sc=%.2f%+.2fi f_brn=%.2f  δ=%8.3f° η=%.4f\n",
            ny, ymax, jacpow, real(f_sc),imag(f_sc), real(f_brn), rad2deg(0.5*angle(S)), abs(S))
    return S
end

println("benchmark doublet 14.1: δ=105.49°, η=0.4649  [Rimas HDR Eq.2.118: scattered(CS)+Born(no CS)]")
# Derived CS Jacobian: jac=e^{2iθ} (x-contour cancels V's e^{−iθ}, y-contour supplies the missing factor).
# Check δ,η at jacpow=2, θ=3°/4°, convergence in mesh + b. Residual magnitude = deuteron c-norm / recoupling?
for θdg in [3.0, 4.0]
    global θ_deg = θdg; global θ = θdg*π/180
    println("=== θ = $(θdg)°, jacpow=2 ===")
    for (nx,ny,xmax,ymax,b) in [(20,70,30.0,100.0,8.0),(24,90,32.0,130.0,8.0),(24,90,32.0,130.0,12.0)]
        run(nx,ny,xmax,ymax,b; jacpow=2.0)
    end
end
