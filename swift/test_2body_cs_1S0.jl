# test_2body_cs_1S0.jl — 2-body isolation test (TODO ▶ NEXT A).
# Single-coordinate complex-scaled MT ¹S₀ two-body scattering, inhomogeneous Lazauskas method.
# Target: Lazauskas-Carbonell PRC 84,034002 (2011) Table I → δ(¹S₀) = 63.512° at E_cm = 1 MeV,
# θ=10°, r_max=100 fm.  η MUST be 1 (single open channel, real potential, no absorption).
#
# Isolates the SCALAR amplitude machinery (c-product vs Hermitian, F_0 normalization, S=1+2ikf
# convention, CS Jacobian) from ALL 3-body recoupling / Rxy. Reuses the proven 2-body builders
# (T_matrix_scaled, Bmatrix, the Lagrange basis); V is built inline forcing s=0 (¹S₀) since
# twobody.jl's V_matrix_scaled hardcodes s=1 (³S₁ deuteron).
using LinearAlgebra, Printf, FastGaussQuadrature
include("../general_modules/mesh.jl");        using .mesh
include("../NNpot/nuclear_potentials.jl");    using .NuclearPotentials
include("laguerre.jl");                       using .Laguerre
include("twobody.jl");                        using .twobodybound
include("coulcc.jl");                         using .CoulCC

const ħ   = 197.3269718
const amu = 931.49432
const mN  = 1.0079713395678829
const μ   = mN/2.0            # 2-body NN reduced mass; ħ²/(2μ) = 41.471

E   = 1.0                     # E_cm in MeV (Lazauskas Table I point)
θ_deg = 10.0; θ = θ_deg*π/180
nx  = 100; xmax = 100.0       # Lazauskas r_max = 100 fm
k   = sqrt(2.0*μ*amu*E)/ħ

# x-only mesh (y unused; keep tiny). initialmesh(nθ,nx,ny,xmax,ymax,alpha)
grid = initialmesh(2, nx, 2, xmax, 2.0, 0.0)

# single-channel ¹S₀ : l=0, s12=0, J12=0
α = twobodybound.nch2b(); α.nchmax = 1; α.s1=0.5; α.s2=0.5
α.l = [0]; α.s12 = [0.0]; α.J12 = 0.0

# --- V (¹S₀, s=0) with CS backward rotation, NO conjugate (mirror V_matrix_scaled) ---
function V_1S0(grid, θ; n_gauss=5*grid.nx)
    nx = grid.nx
    V = zeros(ComplexF64, nx, nx)
    if θ == 0.0
        for ir in 1:nx
            V[ir,ir] = potential_matrix("MT", grid.xi[ir], [0], 0, 0, 0, 0)[1,1]
        end
        return V
    end
    rq, wq = gausslegendre(n_gauss)
    rq = (rq .+ 1.0).*(xmax/2.0); wq = wq.*(xmax/2.0)
    for iq in 1:n_gauss
        r_k = rq[iq]; w_k = wq[iq]
        phi = lagrange_laguerre_regularized_basis(r_k, grid.xi, grid.ϕx, grid.α, grid.hsx, θ)
        vpot = potential_matrix("MT", r_k, [0], 0, 0, 0, 0)[1,1]
        @inbounds for j in 1:nx, i in 1:nx
            V[i,j] += w_k * phi[i]*vpot*phi[j]
        end
    end
    V .*= exp(-im*θ)   # Jacobian
    return V
end

T = twobodybound.T_matrix_scaled(α, grid, θ_deg=θ_deg)
B = ComplexF64.(twobodybound.Bmatrix(α, grid))
V = V_1S0(grid, θ)

# --- source φ_in : regular free wave F_0(k r e^{iθ}) / ϕx  (η=0 → F_0 = sin) ---
φ_in = zeros(ComplexF64, nx)
for i in 1:nx
    arg = ComplexF64(k * grid.xi[i] * exp(im*θ))
    fc, gc, fcp, gcp, sig, ifail = coulcc(arg, ComplexF64(0.0), 0; lmax=0, mode=4)
    φ_in[i] = (ifail==0 ? fc[1] : 0.0im) / grid.ϕx[i]   # fc[1] = F_{λ=0}
end

# --- solve [E B - T - V] c_sc = V φ_in ;  u_total = φ_in + c_sc ---
A = E.*B .- T .- V
b = V * φ_in
c_sc = A \ b
u_tot = φ_in .+ c_sc
@printf("‖A c_sc - b‖/‖b‖ = %.2e\n", norm(A*c_sc - b)/norm(b))

# --- amplitude f = -(2μ/ħ²k²) ⟨F_0|V|u_tot⟩ ; S = 1 + 2ik f ---
M = transpose(φ_in) * (V * u_tot)            # raw c-product matrix element ⟨F_0|V|u_tot⟩
M_sc = transpose(φ_in) * (V * c_sc)          # only scattered part
@printf("\nraw M = ⟨F_0|V|u_tot⟩ (bilinear) = %.5f + %.5f i\n", real(M), imag(M))

# --- COLOSS-EXACT Born/scattered split (scatt.f:54-62): f_born UNROTATED, f_sc ROTATED+e^{iθ} ---
# f = -(1/E)·⟨F|V|F⟩_unrot  -  (e^{iθ}/E)·⟨F_rot|V_rot|c_sc⟩_rot
V_unrot = V_1S0(grid, 0.0)                   # unrotated V (V_origin)
φ_in_unrot = zeros(ComplexF64, nx)           # unrotated regular F_0(k r)
for i in 1:nx
    arg = ComplexF64(k * grid.xi[i])         # NO e^{iθ}
    fc, gc, fcp, gcp, sig, ifail = coulcc(arg, ComplexF64(0.0), 0; lmax=0, mode=4)
    φ_in_unrot[i] = (ifail==0 ? fc[1] : 0.0im) / grid.ϕx[i]
end
M_born_unrot = transpose(φ_in_unrot) * (V_unrot * φ_in_unrot)   # ⟨F|V|F⟩ unrotated
M_sc_rot     = transpose(φ_in) * (V * c_sc)                     # ⟨F_rot|V_rot|c_sc⟩ rotated (=M_sc)
f_coloss = -(1.0/(E)) * M_born_unrot  -  (exp(im*θ)/(E)) * M_sc_rot
# NOTE 1/E = 2μ/(ħ²k²) since k²=2μE/ħ². COLOSS prefactor is exactly -1/ecm.
S_coloss = 1 + 2im*k*f_coloss
@printf("COLOSS-split: f_born(unrot)=%.4f%+.4fi  f_sc(rot)=%.4f%+.4fi\n",
        real(-M_born_unrot/E), imag(-M_born_unrot/E),
        real(-exp(im*θ)*M_sc_rot/E), imag(-exp(im*θ)*M_sc_rot/E))
@printf(">>> COLOSS Born/scattered split : δ = %8.3f°   η = %.5f  (target 63.512°, 1)\n",
        rad2deg(0.5*angle(S_coloss)), abs(S_coloss))

# target: δ=63.512° → tan δ = %.4f, and e^{iδ}sinδ = %.4f%+.4fi
δt = deg2rad(63.512); println("target tanδ = ", round(tan(δt),digits=4),
      "   e^{iδ}sinδ = ", round(cos(δt)*sin(δt),digits=4), " + ", round(sin(δt)^2,digits=4), "i")

println("="^66)
@printf("Lazauskas Table I  MT ¹S₀  E_cm=1 MeV, θ=%.0f°, r_max=%.0f : δ=63.512°, η=1\n", θ_deg, xmax)
@printf("k = %.5f fm⁻¹\n", k)
# VERDICT: the correct CS amplitude is f = -2μ/(ħ²k²)·e^{+iθ}·⟨F_0|V|u_tot⟩ (bilinear c-product).
# The e^{+iθ} is the CS contour Jacobian (one factor per rotated radial integration). It restores
# η→1 (unitarity) and reproduces δ=63.512° to CS-convergence. WITHOUT it η≈1.16 (the layer-2 bug).
f_correct = -2.0*μ*amu/(ħ^2*k^2) * exp(im*θ) * M
Sc = 1 + 2im*k*f_correct
@printf(">>> CORRECT  f=-2μ/(ħ²k²)·e^{iθ}·M : δ = %8.3f°   η = %.5f  (target 63.512°, 1)\n",
        rad2deg(0.5*angle(Sc)), abs(Sc))
# candidate amplitude conventions (T-matrix S=1+2ikf):
for (name, f) in [
        ("-2μ/(ħ²k²)·M",      -2.0*μ*amu/(ħ^2*k^2)*M),
        ("-2μ/(ħ²k) ·M  (×k)", -2.0*μ*amu/(ħ^2*k)*M),
        ("-2μ/(ħ²k³)·M  (/k)", -2.0*μ*amu/(ħ^2*k^3)*M),
    ]
    S = 1 + 2im*k*f
    @printf("  %-22s : δ = %8.3f°   η = %.5f\n", name, rad2deg(0.5*angle(S)), abs(S))
end
println("-"^66)
println("phase-factor scan on f = -2μ/(ħ²k²)·M·e^{inθ}  (|f| already matches; isolate the CS phase):")
f0 = -2.0*μ*amu/(ħ^2*k^2)*M
for n in [-2.0,-1.0,-0.5,0.5,1.0,2.0]
    f = f0*exp(im*n*θ); S = 1+2im*k*f
    @printf("  n=%+.1f (e^{%+.0fθ}) : δ = %8.3f°   η = %.5f\n", n, n, rad2deg(0.5*angle(S)), abs(S))
end
println("="^66)
