# test_scatt_diag2.jl — route H1 (extraction) vs H2 (generation) in ONE reduced-mesh solve.
# Checks, with NO amplitude formula:
#   (A6) y-profile of ψ_in and of the source b = 2·V·Rxy·ψ_in : does ψ_in decay/grow? is b localized?
#   (A5/B5) y-profile of ψ_sc (should decay ~exp(-q sinθ y) if CS is working) and the breakup weight
#           W_brk = Σ_{non-deuteron channels} ⟨ψ_sc|B|ψ_sc⟩ vs deuteron-channel weight.
using LinearAlgebra, Printf
include("../general_modules/channels.jl"); using .channels
include("../general_modules/mesh.jl");     using .mesh
include("matrices_optimized.jl");          using .matrices_optimized
include("scattering.jl");                   using .Scattering
include("twobody.jl");                      using .twobodybound

const ħ=197.3269718; const m=1.0079713395678829; const amu=931.49432

fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
nx=12; ny=30; xmax=24.0; ymax=50.0; nθ=12; alpha=1.0
θ_deg=10.0; θ=θ_deg*π/180
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0

α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)

bE,bψ = bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
E_total=E_cm+E_d
ψ_in = compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
ψ_sc,A,b = solve_scattering_equation(E_total,α,grid,potname,ψ_in,θ_deg=θ_deg)

μ=2m/3; q=sqrt(2*μ*amu*E_cm)/ħ
@printf("\nE_cm=%.3f E_d=%.4f E_total=%.3f  q=%.4f fm^-1  q*sinθ=%.4f\n",E_cm,E_d,E_total,q,q*sin(θ))

# helper: y-profile of a state vector for a given channel block
yprof(v,iα) = [sqrt(sum(abs2, v[(iα-1)*nx*ny+(ix-1)*ny+iy] for ix in 1:nx)) for iy in 1:ny]

# deuteron channels (J12=1) vs non-deuteron (breakup-like)
deut = [iα for iα in 1:α.nchmax if Int(round(α.α2b.J12[α.α2bindex[iα]]))==1]
nond = setdiff(1:α.nchmax, deut)
println("deuteron(J12=1) channels: ",deut,"   non-deuteron(breakup) channels: ",nond)

# pick the elastic ³S₁,λ=0 channel for the profile
elastic = first([iα for iα in deut if α.α2b.l[α.α2bindex[iα]]==0 && Int(round(α.λ[iα]))==0])
@printf("\n--- y-profile (elastic ³S₁λ0 channel %d), sampled ---\n", elastic)
pin=yprof(ψ_in,elastic); pb=yprof(b,elastic); psc=yprof(ψ_sc,elastic)
@printf("%4s %12s %12s %12s %14s\n","iy","y","|ψ_in|","|b src|","|ψ_sc|")
for iy in 1:max(1,ny÷10):ny
    @printf("%4d %12.3f %12.3e %12.3e %14.3e\n", iy, grid.yi[iy], pin[iy], pb[iy], psc[iy])
end
@printf("ratios last/first:  ψ_in %.2e   b %.2e   ψ_sc %.2e   (CS-correct: ψ_sc≪1, b localized)\n",
        pin[end]/pin[1], pb[end]/pb[1], psc[end]/psc[1])

# breakup weight test (A5/B5) — per-channel overlap metric Bxy = Nx ⊗ Ny
Nx = matrices_optimized.compute_overlap_matrix(grid.nx, grid.xx)
Ny = matrices_optimized.compute_overlap_matrix(grid.ny, grid.yy)
Bxy = kron(Nx, Ny)
chw(v,chans)= real(sum( (v[(iα-1)*nx*ny+1:iα*nx*ny])' * Bxy * (v[(iα-1)*nx*ny+1:iα*nx*ny]) for iα in chans))
Wd=chw(ψ_sc,deut); Wb=chw(ψ_sc,nond)
@printf("\n--- breakup content of ψ_sc ---\n")
@printf("deuteron-channel weight   ⟨ψ_sc|B|ψ_sc⟩_deut = %.4e\n",Wd)
@printf("non-deuteron(breakup) wt  ⟨ψ_sc|B|ψ_sc⟩_brk  = %.4e\n",Wb)
@printf("breakup fraction W_brk/(W_brk+W_d) = %.4f   (≈0 → H2 generation bug; sizable → H1 extraction)\n",
        Wb/(Wb+Wd))
