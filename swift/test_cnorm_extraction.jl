# test_cnorm_extraction.jl — decisive test of the amplitude inner-product convention.
# ONE reduced-mesh doublet n-d solve (E_lab=14.1, θ=10°), then extract the elastic
# S-matrix TWO ways from the SAME ψ_sc:
#   (old)  conj_bra=true  : dot()  = Hermitian/conjugated bra  → expected η≈1 (wrong)
#   (new)  conj_bra=false : transpose() = bilinear c-product   → expected η↓ toward 0.4649
# Deuteron now carries the complex CS c-norm C_n (twobody.jl fix). Benchmark: δ=105.49°, η=0.4649.
using LinearAlgebra, Printf
include("../general_modules/channels.jl"); using .channels
include("../general_modules/mesh.jl");     using .mesh
include("matrices_optimized.jl");          using .matrices_optimized
include("scattering.jl");                   using .Scattering
include("twobody.jl");                      using .twobodybound

const ħ=197.3269718; const m=1.0079713395678829; const amu=931.49432

fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
nx=30; ny=70; xmax=30.0; ymax=60.0; nθ=12; alpha=1.0   # benchmark-converged mesh
θ_deg=10.0; θ=θ_deg*π/180
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0

α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)

V           = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
bE,bψ = bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
@printf("  E_d = %.4f MeV  (deuteron Hermitian-normalized, bound-state code untouched)\n", E_d)

E_total=E_cm+E_d
ψ_in = compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
ψ_sc,A,b = solve_scattering_equation(E_total,α,grid,potname,ψ_in,θ_deg=θ_deg)
@printf("rel_res = %.2e\n", norm(A*ψ_sc-b)/norm(b))

μ=2m/3; k=sqrt(2*μ*amu*E_cm)/ħ

# Deuteron complex c-norm  C_n = φ^T B_2b φ  (φ is Hermitian-normalized, so |C_n|≈1 but phase≠0).
# Reconstruct the 2-body eigenvector from the wavefunction matrix φ_d[j,ich]=ϕx[j]*evec[idx].
n2b = size(φ_d,2)
evec = ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
Ix = [ (i==j ? 1.0 : 0.0) + (-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
B2b = kron(Matrix{Float64}(I,n2b,n2b), Ix)
C_n = transpose(evec)*B2b*evec
@printf("\nDeuteron c-norm  C_n = %.4f + %.4f i   |C_n|=%.4f  arg=%.2f°\n",
        real(C_n), imag(C_n), abs(C_n), rad2deg(angle(C_n)))

function delta_eta(conj_bra; cfac=1.0)
    f,dch,_ = compute_scattering_amplitude(ψ_in,V,Rxy_31,ψ_sc,E_cm,grid,α,φ_d,z1z2;
                                           θ=θ, σ_l=0.0, conj_bra=conj_bra)
    f = f .* cfac
    U = compute_collision_matrix(f,k)
    U_cs,lab = Scattering.recouple_to_channel_spin(U,α,dch)
    s = U_cs[(Jtot,1)][findfirst(==("λ=0, 𝕊=0.5"), lab[(Jtot,1)])]
    return rad2deg(0.5*angle(s)), abs(s)
end

δ_old,η_old   = delta_eta(true)

println("\n"*"="^64)
@printf("benchmark (Lazauskas Tab.III doublet 14.1): Re δ=105.49°, η=0.4649\n")
@printf("OLD  conj bra (dot)            : Re δ = %8.3f°   η = %.4f\n", δ_old, η_old)
println("-"^64)
println("CS-Jacobian scan: bilinear × 1/C_n × e^{inθ}  (2-body needed n=1 per radial integration)")
for n in 0.0:1.0:4.0
    δ,η = delta_eta(false; cfac = exp(im*n*θ)/C_n)
    flag = abs(η-0.4649)<0.05 ? "  <<< η match" : ""
    @printf("  1/C_n × e^{%.0fθ} : Re δ = %8.3f°   η = %.4f%s\n", n, δ, η, flag)
end
println("="^64)

# ======================================================================
# Lazauskas Eq.16 asymptotic projection (VALIDATION of direction; Eq.17 is the
# production method per Lazauskas).  f_λ = C_n^{-1} y e^{-iq y e^{iθ}} ∫φ_d(xe^{-iθ})ψ_sc(x,y)e^{3iθ}d³x,
# plateau over large y.  The e^{3iθ} cancels between numerator and C_n (both bare overlaps).
# Project the ELASTIC λ=0 ³S₁ deuteron channel.
println("\n"*"="^64)
println("Lazauskas Eq.16 asymptotic y-projection (validation, elastic λ=0 ³S₁):")
nx=grid.nx; ny=grid.ny
Nx = matrices_optimized.compute_overlap_matrix(nx, grid.xx)
# elastic 3-body channel: J12=1, l=0 (³S₁), λ=0
elastic = first([iα for iα in 1:α.nchmax if Int(round(α.α2b.J12[α.α2bindex[iα]]))==1 &&
                 α.α2b.l[α.α2bindex[iα]]==0 && Int(round(α.λ[iα]))==0])
φd_S = φ_d[:,1] ./ grid.ϕx          # ³S₁ deuteron coefficient (value/ϕx)
# ψ_sc coefficients for the elastic channel, as [nx, ny]
csc = [ ψ_sc[(elastic-1)*nx*ny + (ix-1)*ny + iy] for ix in 1:nx, iy in 1:ny ]
q = k
fofy = ComplexF64[]
for iy in 1:ny
    P = transpose(φd_S) * Nx * csc[:,iy]          # bilinear x-overlap (bare; e^{3iθ} cancels vs C_n)
    y = grid.yi[iy]
    push!(fofy, (1/C_n) * y * exp(-im*q*y*exp(im*θ)) * P)
end
@printf("%4s %10s %14s %14s\n","iy","y","Re f_λ(y)","Im f_λ(y)")
for iy in ny-9:ny
    @printf("%4d %10.3f %14.4e %14.4e\n", iy, grid.yi[iy], real(fofy[iy]), imag(fofy[iy]))
end
fpl = fofy[ny-2]                                   # plateau value (near-but-not-edge node)
for S in [1+2im*fpl, 1+2im*q*fpl, 1+2im*fpl/q]
    @printf("  S=1+2i·(var)f : δ=%8.3f°  η=%.4f   (|f_pl|=%.3e)\n",
            rad2deg(0.5*angle(S)), abs(S), abs(fpl))
end
println("="^64)
