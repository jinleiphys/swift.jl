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

function build(nx,ny,xmax,ymax,b)
    α=α3b(fermion,Jtot,T,Parity,2,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    bE,bψ=bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    A,B,Tm,V,Rxy,Rxy_31,_,_,_,Nx,Ny,_ = Scattering.compute_scattering_matrix(E_cm+E_d,α,grid,potname,θ_deg=θ_deg)
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
    # c-norm (bilinear, e^{iθ})
    n2b=size(φ_d,2); evec=ComplexF64[φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx]
    Ix=[(i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx,j in 1:grid.nx]
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)
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
    return (;α,grid,A,B,V,Rxy,q,C_n,λs,chans,nxg,nyg,blk,Omega_R,Omega_I_and_d3)
end

function run(nx,ny,xmax,ymax,b; jacpow=1.0, useCn=true)
    s=build(nx,ny,xmax,ymax,b); λs=s.λs; nc=length(λs)
    VRxy(u)=s.V*(s.Rxy*u)
    ΩR=Dict(λ=>s.Omega_R(λ) for λ in λs)
    UR=Dict(); UI=Dict()
    for λ in λs
        vR=s.A\VRxy(ΩR[λ]); UR[λ]=ΩR[λ].+vR
        Ω̃,d3=s.Omega_I_and_d3(λ); vI=s.A\(s.B*d3 .+ VRxy(Ω̃)); UI[λ]=Ω̃.+vI
    end
    jac=exp(im*jacpow*θ); cn= useCn ? s.C_n : 1.0+0im
    function Mel(c_bra, U)   # ⟨Ω_c^(R)|V·Rxy|U⟩ summed over channel-c's swift-channels
        W=VRxy(U); acc=0.0im
        for iα in s.chans[λs[c_bra]]; acc+=transpose(s.blk(ΩR[λs[c_bra]],iα))*s.blk(W,iα); end
        return (jac/cn)*acc
    end
    MR=[Mel(c, UR[λs[c0]]) for c in 1:nc, c0 in 1:nc]
    MI=[Mel(c, UI[λs[cc]]) for c in 1:nc, cc in 1:nc]
    Amat=[ (ħ2_2μy*s.q)*(c==cc ? 1.0 : 0.0)+MI[c,cc] for c in 1:nc, cc in 1:nc]
    @printf("    MR=[%.2f%+.2fi %.2f%+.2fi; %.2f%+.2fi %.2f%+.2fi] MI=[%.1f%+.1fi %.1f%+.1fi; %.1f%+.1fi %.1f%+.1fi] ℏ²q/2µ=%.2f\n",
            real(MR[1,1]),imag(MR[1,1]),real(MR[1,2]),imag(MR[1,2]),real(MR[2,1]),imag(MR[2,1]),real(MR[2,2]),imag(MR[2,2]),
            real(MI[1,1]),imag(MI[1,1]),real(MI[1,2]),imag(MI[1,2]),real(MI[2,1]),imag(MI[2,1]),real(MI[2,2]),imag(MI[2,2]), ħ2_2μy*s.q)
    K = -(Amat\MR)                      # solve A·K = −MR (columns)
    @printf("    K=[%.3f%+.3fi  %.3f%+.3fi; %.3f%+.3fi  %.3f%+.3fi] | ‖U_λ2‖=%.2e MR22=%.3f%+.3fi\n",
            real(K[1,1]),imag(K[1,1]),real(K[1,2]),imag(K[1,2]),real(K[2,1]),imag(K[2,1]),real(K[2,2]),imag(K[2,2]),
            norm(UR[λs[2]]), real(MR[2,2]),imag(MR[2,2]))
    Smat=(I(nc)+im*K)*inv(I(nc)-im*K)
    ev=eigen(Smat); evals=ev.values; evecs=ev.vectors
    id=argmax(abs.(evecs[1,:]))         # λ=0-dominant = doublet
    sd=evals[id]; so=evals[3-id>0 ? (id==1 ? 2 : 1) : id]
    @printf("  jac e^{%.1fθ} Cn=%s b=%.1f : doublet δ=%8.3f° η=%.4f | other δ=%8.3f° η=%.4f\n",
            jacpow, useCn, b, rad2deg(0.5*angle(sd)),abs(sd), rad2deg(0.5*angle(so)),abs(so))
end
println("benchmark doublet 14.1: δ=105.49°, η=0.4649")
for jp in [0.0,1.0,2.0]
    run(16,40,28.0,60.0, 8.0; jacpow=jp)
end
