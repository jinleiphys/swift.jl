# test_eq17_amplitude.jl — full Lazauskas Eq.17 (Green's theorem) n+d elastic amplitude.
#   f = -C_n^{-1}(m/ħ²) ∬ φ_d*(xe^{-iθ}) ĥ⁻_λ(qy e^{iθ}) [V₂₃+V₃₁] Ψ̄ e^{6iθ} d³x d³y
# with the validated operator  [V₂₃+V₃₁]Ψ̄ = Rxy·V·Ψ̄,  Ψ̄=(I+Rxy)ψ_total, ψ_total=ψ_in+ψ_sc.
# bra y-part = incoming Riccati-Hankel ĥ⁻_λ = G − iF (so the bilinear product with the outgoing
# scattered wave projects the amplitude).  Scan the overall f→S convention constant against the
# benchmark (Lazauskas Tab.III doublet 14.1: Re δ=105.49°, η=0.4649).
using LinearAlgebra, Printf
include("../general_modules/channels.jl"); using .channels
include("../general_modules/mesh.jl");     using .mesh
include("matrices_optimized.jl");          using .matrices_optimized
include("scattering.jl");                   using .Scattering
include("twobody.jl");                      using .twobodybound
include("coulcc.jl");                       using .CoulCC

const ħ=197.3269718; const m=1.0079713395678829; const amu=931.49432

fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
nx=12; ny=30; xmax=24.0; ymax=50.0; nθ=12; alpha=1.0
θ_deg=10.0; θ=θ_deg*π/180
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0

α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)
V           = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
bE,bψ = bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
E_total=E_cm+E_d
ψ_in = compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
ψ_sc,A,b = solve_scattering_equation(E_total,α,grid,potname,ψ_in,θ_deg=θ_deg)
μ=2m/3; q=sqrt(2*μ*amu*E_cm)/ħ
@printf("rel_res=%.2e  q=%.4f\n", norm(A*ψ_sc-b)/norm(b), q)

# --- deuteron c-norm C_n (bilinear, includes e^{3iθ}) ---
n2b=size(φ_d,2)
evec=ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
Ix=[ (i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
C_n = (transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec) * exp(3im*θ)

# --- operator (CORRECTED 2026-06-16): post-form amplitude ⟨ψ_in| V·Rxy |ψ_total⟩.
#     V·Rxy is the SAME operator that builds the source b=2V·Rxy_31·φ (Rxy=2Rxy_31), acting on
#     the Faddeev component ψ_total=ψ_in+ψ_sc. The diagnostic test_eq17_magnitude.jl showed this
#     gives η=0.471≈benchmark 0.4649, whereas Rxy·V·Ψ̄ (the bound-state expectation operator) is
#     off by ×58. The earlier "[V₂₃+V₃₁]Ψ̄ = Rxy·V·Ψ̄" identity is NOT the scattering amplitude. ---
ψ_total = ψ_in .+ ψ_sc
W       = V*(Rxy*ψ_total)

# --- bra = REGULAR free wave φ_d(x)·F_λ(qy)/(ϕx ϕy) (thesis Eq.1.42 uses regular ĵ_l, NOT Hankel;
#     matches the validated 2-body test). This is exactly ψ_in. ---
nxg=grid.nx; nyg=grid.ny; nch=α.nchmax
deut=Int[]; labels=String[]
for iα in 1:nch
    i2b=α.α2bindex[iα]
    if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
        l=α.α2b.l[i2b]; mc= l==0 ? 1 : (l==2 ? 2 : 0)
        if mc>0 && norm(φ_d[:,mc])>1e-8
            push!(deut,iα); push!(labels,"λ=$(Int(round(α.λ[iα]))), 𝕊=0.5")
        end
    end
end
nd=length(deut)
blk(v,iα)= v[(iα-1)*nxg*nyg+1 : iα*nxg*nyg]

# matrix element M[out,in] = ψ_in(out)ᵀ · W(in)  (bilinear; W = Rxy·V·Ψ̄ = [V₂₃+V₃₁]Ψ̄)
M=zeros(ComplexF64,nd,nd)
for io in 1:nd, ii in 1:nd
    M[io,ii]= transpose(blk(ψ_in,deut[io])) * blk(W,deut[ii])
end

# prefactor −2μ/(ħ²k²) (2-body-anchored form, μ = n-d reduced mass) × 1/C_n × jacobian e^{inθ};
# 2-body fixed e^{iθ} per radial integration → scan n for the 3-body (x,y) amplitude. S = 1+2iq·f.
println("benchmark doublet 14.1: Re δ=105.49°, η=0.4649   (CORRECT operator Rxy·V·Ψ̄ + regular F bra)")
@printf("raw |M[1,1]|=%.3e\n", abs(M[1,1]))
pref0 = -2.0*μ*amu/(ħ^2*q^2) / C_n
for n in 0.0:1.0:6.0
    f0 = (pref0*exp(im*n*θ)) .* M
    U = compute_collision_matrix(f0, q)
    Ucs,lab = Scattering.recouple_to_channel_spin(U, α, deut)
    key=(Jtot,1); L=lab[key]; idx=findfirst(==("λ=0, 𝕊=0.5"),L)
    s = Ucs[key][idx,idx]
    flag = abs(abs(s)-0.4649)<0.05 ? "  <<< η" : ""
    @printf("  jac e^{%.0fθ} : Re δ = %8.3f°   η = %.4f%s\n", n, rad2deg(0.5*angle(s)), abs(s), flag)
end
