# test_eq17_magnitude.jl — localize the phase-INDEPENDENT magnitude factor (~58) in the
# n+d elastic Eq.17 amplitude. The Jacobian e^{inθ} only rotates δ (global phase), so the
# magnitude error must be in the OPERATOR/KET choice or a clean measure factor.
# Target (Lazauskas Tab.III doublet 14.1): η=0.4649, Re δ=105.49°  ⟹  |S−1|=1.417, |f|=1.289.
# Bra = regular F (=ψ_in) per thesis Eq.1.63 + validated 2-body test_2body_cs_1S0.jl.
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

# deuteron c-norm (bilinear)
n2b=size(φ_d,2)
evec=ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
Ix=[ (i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
C_n = (transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec) * exp(im*θ)   # NOTE e^{1iθ}: 1D radial Jacobian (NOT e^{3iθ})

ψ_total = ψ_in .+ ψ_sc
Ψbar    = ψ_total .+ Rxy*ψ_total

# elastic deuteron λ=0 channel index
nxg=grid.nx; nyg=grid.ny; nch=α.nchmax
deut=Int[]
for iα in 1:nch
    i2b=α.α2bindex[iα]
    if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
        l=α.α2b.l[i2b]; mc= l==0 ? 1 : (l==2 ? 2 : 0)
        if mc>0 && norm(φ_d[:,mc])>1e-8; push!(deut,iα); end
    end
end
iel = first(deut)
blk(v,iα)= v[(iα-1)*nxg*nyg+1 : iα*nxg*nyg]

target_S1 = 1.417   # |S−1| for benchmark (η=0.4649, δ=105.49°)
target_f  = 1.289
pref = -2.0*μ*amu/(ħ^2*q^2) / C_n        # |pref| = 1/E_cm/|C_n|
@printf("\n|pref| = %.4f  (1/E_cm=%.4f, |C_n|=%.4f)\n", abs(pref), 1/E_cm, abs(C_n))
@printf("target |f| = %.3f, |S−1| = %.3f\n", target_f, target_S1)
println("="^78)
@printf("%-34s  %10s  %8s  %8s  %8s\n","operator  ⟨ψ_in| · |·⟩ (elastic)","|M|","|f|","η","f/ftgt")
println("-"^78)

function report(name, ket)
    M = transpose(blk(ψ_in,iel)) * blk(ket, iel)
    f = pref * M
    S = 1 + 2im*q*f
    @printf("%-34s  %10.3e  %8.3f  %8.3f  %8.2f\n", name, abs(M), abs(f), abs(S), abs(f)/target_f)
end

report("Rxy·V·Ψ̄   (current)",        Rxy*(V*Ψbar))
report("V·Ψ̄",                         V*Ψbar)
report("Rxy·V·ψ_total",               Rxy*(V*ψ_total))
report("V·ψ_total",                   V*ψ_total)
report("2·V·Rxy_31·ψ_total (src-op)", 2 .*(V*(Rxy_31*ψ_total)))
report("V·Rxy·ψ_total",               V*(Rxy*ψ_total))
report("Rxy·V·ψ_sc",                  Rxy*(V*ψ_sc))
report("V·ψ_sc",                      V*ψ_sc)
println("="^78)

# WINNER: V·Rxy·ψ_total gives η≈0.47 ≈ benchmark. Scan Jacobian e^{inθ} to also hit δ=105.49°.
println("\n>>> WINNER operator V·Rxy·ψ_total : Jacobian-phase scan to match δ=105.49°, η=0.4649")
ket = V*(Rxy*ψ_total)
M0  = transpose(blk(ψ_in,iel)) * blk(ket, iel)
for n in 0.0:0.5:6.0
    f = pref * exp(im*n*θ) * M0
    S = 1 + 2im*q*f
    flag = (abs(abs(S)-0.4649)<0.03 && abs(rad2deg(0.5*angle(S))-105.49)<8) ? "  <<<" : ""
    @printf("  jac e^{%.1fθ}: Re δ = %8.3f°   η = %.4f%s\n", n, rad2deg(0.5*angle(S)), abs(S), flag)
end
println("benchmark: Re δ = 105.490°,  η = 0.4649")
