# test_V23V31_operator.jl — validate the [V₂₃+V₃₁] operator construction (Eq.17 step 1).
#
# Independent check (NO fitting): for a 3-body eigenstate Ψ̄, Schrödinger gives
#   V_total Ψ̄ = (E·B - T) Ψ̄,  V_total = V₁₂ + [V₂₃+V₃₁]
#   ⟹  [V₂₃+V₃₁] Ψ̄ = (E·B - T - V₁₂) Ψ̄        (RHS uses NO Rxy — the independent ruler)
# Build [V₂₃+V₃₁]Ψ̄ several ways via the Rxy rearrangements and see which matches the
# Schrödinger RHS to truncation level (few %). Wrong Rxy ordering → O(1) mismatch.
using LinearAlgebra, Printf
include("../general_modules/channels.jl"); using .channels
include("../general_modules/mesh.jl");     using .mesh
include("matrices_optimized.jl");          using .matrices_optimized
include("twobody.jl");                      using .twobodybound
include("MalflietTjon.jl");                 import .MalflietTjon: malfiet_tjon_solve_optimized

fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
s1=0.5;s2=0.5;s3=0.5;t1=0.5;t2=0.5;t3=0.5
lmax=2; lmin=0; λmax=4; λmin=0; j2bmax=1.0
nθ=12; nx=16; ny=16; xmax=16.0; ymax=16.0; alpha=1.0
potname="MN"

α   = α3b(fermion,Jtot,T,Parity,lmax,lmin,λmax,λmin,s1,s2,s3,t1,t2,t3,MT,j2bmax)
grid= initialmesh(nθ,nx,ny,xmax,ymax,Float64(alpha))

e2b,_ = bound2b(grid,potname)
result, Ψbar, ψ3 = malfiet_tjon_solve_optimized(α,grid,potname,e2b,
        E0=-8.5,E1=-7.5,tolerance=1e-6,max_iterations=30,verbose=false,include_uix=false)
E = real(result.energy)
@printf("\n3-body bound state E = %.5f MeV   (Ψ̄ = full wave, ψ3 = Faddeev component)\n", E)

# matrices (θ=0, real)
T, = T_matrix_optimized(α, grid)
V  = V_matrix_optimized(α, grid, potname)
Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
Rxy_13      = Rxy_13_matrix_optimized(α, grid)
Nx = matrices_optimized.compute_overlap_matrix(nx, grid.xx)
Ny = matrices_optimized.compute_overlap_matrix(ny, grid.yy)
B  = kron(Matrix{Float64}(I,α.nchmax,α.nchmax), kron(Nx, Ny))

Ψ = ComplexF64.(Ψbar)

# T-FREE ruler (avoids the high-momentum noise of T·Ψ̄): by identical-particle symmetry
#   ⟨Ψ̄|V₂₃|Ψ̄⟩ = ⟨Ψ̄|V₃₁|Ψ̄⟩ = ⟨Ψ̄|V₁₂|Ψ̄⟩  ⟹  ⟨Ψ̄|[V₂₃+V₃₁]|Ψ̄⟩ = 2⟨Ψ̄|V₁₂|Ψ̄⟩
VΨ   = V*Ψ
eref = 2 * real(Ψ' * VΨ)                       # = 2⟨V₁₂⟩, the ruler
@printf("\nruler 2⟨Ψ̄|V₁₂|Ψ̄⟩ = %.5e\n", eref)

# candidate vectors for [V₂₃+V₃₁]Ψ̄, and their expectation ⟨Ψ̄|·⟩ vs the ruler
R31=Rxy_31
cands = Dict(
 "Rxy·V·Ψ̄  (DERIVED (P⁺+P⁻)V₁₂Ψ̄)" => Rxy*VΨ,
 "V·Rxy·Ψ̄  (wrong order)"          => V*(Rxy*Ψ),
 "2·R31·V·R13·Ψ̄ (Rxy·V·Rxy-type)"  => 2 .*(R31*(V*(Rxy_13*Ψ))),
)
println("--- expectation check  ⟨Ψ̄|cand⟩ / (2⟨V₁₂⟩)  (=1.000 ⇒ correct) ---")
for (name,c) in cands
    @printf("  %-34s : ⟨Ψ̄|cand⟩/ruler = %.4f\n", name, real(Ψ'*c)/eref)
end

# symmetry sanity (basis of the derivation):  (P⁺+P⁻)Ψ̄ = 2Ψ̄  ⟺  Rxy·Ψ̄ = 2Ψ̄
@printf("\nsymmetry check ‖Rxy·Ψ̄ − 2Ψ̄‖/‖2Ψ̄‖ = %.4f   (small ⇒ P⁺Ψ̄=Ψ̄ holds; same truncation as 0.956)\n",
        norm(Rxy*Ψ .- 2 .*Ψ)/norm(2 .*Ψ))
