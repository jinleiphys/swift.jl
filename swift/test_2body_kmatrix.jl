# test_2body_kmatrix.jl — 2-body K-matrix integral-relations method (Duerinck thesis App A.10-A.12),
# CS MT ¹S₀, anchored on test_2body_cs_1S0.jl (δ=63.512°, η=1). Locks the K-matrix METHOD before 3-body.
#
# Method:  u_l = u^(R) + K_l u^(I),  u^(R)=φ^(R)+F_l,  u^(I)=φ^(I)+G̃_l,  G̃_l=G_l·(1−e^{−r/b})^{2l+1}.
#   (H̄0+V−E)φ^(R) = −V·F_l                       [since (H̄0−E)F_l=0]
#   (H̄0+V−E)φ^(I) = −[(H̄0−E)G̃_l + V·G̃_l]        [(H̄0−E)G̃_l = localized regularization defect]
# In swift's basis A=E·B−T−V represents (E−H̄0−V):  A·φ^(R)=V·F_l ; A·φ^(I)=d + V·G̃_l, d=(H̄0−E)G̃_l.
# K_l = −∫F_l V u^(R) / [ℏ²k/2µ + ∫F_l V u^(I)]   (A.12, V_C=0);  S=(1+iK)/(1−iK), δ=atan(K), η=|S|.
# l=0 defect under CS (H̄0^θ=e^{−2iθ}H̄0):  d(z)=−E[2 G_0'(z) reg_z' + G_0(z) reg_z''],  z=kr e^{iθ},
#   reg_z=1−e^{−z/(kb)},  reg_z'=(1/(kb))e^{−z/(kb)},  reg_z''=−(1/(kb))²e^{−z/(kb)}.
using LinearAlgebra, Printf, FastGaussQuadrature
const BB="/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/general_modules/mesh.jl");        using .mesh
include(BB*"/NNpot/nuclear_potentials.jl");    using .NuclearPotentials
include(BB*"/swift/laguerre.jl");              using .Laguerre
include(BB*"/swift/twobody.jl");               using .twobodybound
include(BB*"/swift/coulcc.jl");                using .CoulCC
const ħ=197.3269718; const amu=931.49432; const mN=1.0079713395678829; const μ=mN/2.0
E=1.0; θ_deg=10.0; θ=θ_deg*π/180; nx=100; xmax=100.0
k=sqrt(2.0*μ*amu*E)/ħ; ħ2_2μ=ħ^2/(2*μ*amu)     # = 41.471 MeV·fm²

grid=initialmesh(2,nx,2,xmax,2.0,0.0)
α=twobodybound.nch2b(); α.nchmax=1; α.s1=0.5; α.s2=0.5; α.l=[0]; α.s12=[0.0]; α.J12=0.0
function V_1S0(grid,θ; n_gauss=5*grid.nx)
    nx=grid.nx; V=zeros(ComplexF64,nx,nx)
    rq,wq=gausslegendre(n_gauss); rq=(rq.+1.0).*(xmax/2.0); wq=wq.*(xmax/2.0)
    for iq in 1:n_gauss
        r_k=rq[iq]; w_k=wq[iq]
        phi=lagrange_laguerre_regularized_basis(r_k,grid.xi,grid.ϕx,grid.α,grid.hsx,θ)
        vpot=potential_matrix("MT",r_k,[0],0,0,0,0)[1,1]
        @inbounds for j in 1:nx,i in 1:nx; V[i,j]+=w_k*phi[i]*vpot*phi[j]; end
    end
    V .*= exp(-im*θ); return V
end
T=twobodybound.T_matrix_scaled(α,grid,θ_deg=θ_deg)
B=ComplexF64.(twobodybound.Bmatrix(α,grid))
V=V_1S0(grid,θ); A=E.*B .- T .- V

# F_0, G_0, G_0' on rotated mesh
F=zeros(ComplexF64,nx); G0=zeros(ComplexF64,nx); G0p=zeros(ComplexF64,nx)
for i in 1:nx
    z=ComplexF64(k*grid.xi[i]*exp(im*θ))
    fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=0,mode=1)  # mode=1 → F,G,F',G'
    F[i]=(ifail==0 ? fc[1] : 0.0im)/grid.ϕx[i]
    G0[i]=(ifail==0 ? gc[1] : 0.0im)            # raw (no /ϕ yet)
    G0p[i]=(ifail==0 ? gcp[1] : 0.0im)          # G' w.r.t. argument z
end

println("benchmark MT ¹S₀: δ=63.512°, η=1  → K=tan δ=", round(tan(deg2rad(63.512)),digits=4))
@printf("k=%.5f ħ²k/2µ=%.4f\n", k, ħ2_2μ*k)
for b in [1.5,2.0,3.0,5.0,8.0,12.0]
    kb=k*b
    G̃=zeros(ComplexF64,nx); dvec=zeros(ComplexF64,nx)
    for i in 1:nx
        z=ComplexF64(k*grid.xi[i]*exp(im*θ)); e=exp(-z/kb)
        regz=1-e; regzp=e/kb; regzpp=-e/kb^2
        G̃[i]=(G0[i]*regz)/grid.ϕx[i]
        dz=-E*(2*G0p[i]*regzp + G0[i]*regzpp)          # (H̄0^θ−E)G̃ = localized defect
        dvec[i]=dz/grid.ϕx[i]
    end
    φR=A\(V*F)            # A φ^(R)=V·F
    φI=A\(B*dvec .+ V*G̃)  # A φ^(I)=d + V·G̃
    uR=F.+φR; uI=G̃.+φI
    jac=exp(im*θ)
    IR=jac*transpose(F)*(V*uR); II=jac*transpose(F)*(V*uI)
    K=-IR/(ħ2_2μ*k + II)
    S=(1+im*K)/(1-im*K)
    # cross-check: test_2body amplitude f=−(1/E)e^{iθ}⟨F|V|uR⟩ (uR=u_tot), K_f=kf/(1+ikf) should be real≈tanδ
    f_amp=-(1.0/E)*jac*transpose(F)*(V*uR); K_f=k*f_amp/(1+im*k*f_amp)
    @printf("  b=%.2f : K=%.3f%+.3fi δ=%7.3f° η=%.4f | I_R=%.3f%+.3fi I_I=%.3f%+.3fi −iI_R=%.3f%+.3fi | K_f(amp)=%.3f%+.3fi\n",
            b, real(K),imag(K), rad2deg(0.5*angle(S)), abs(S),
            real(IR),imag(IR),real(II),imag(II),real(-im*IR),imag(-im*IR), real(K_f),imag(K_f))
end
