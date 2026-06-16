# test_b1_b5_diag.jl — idea-pk diagnostics B1 (Rxy metric-adjoint audit) + B5 (channel-spin projection).
# B1: the analytic identity ⟨bra|V₂₃+V₃₁|Ψ̄⟩=(Rxy·bra)ᵀVΨ̄ assumed Rxyᵀ=Rxy. In the non-orthogonal
#     Laguerre metric B that is FALSE. Compare 4 rearrangement maps of the bra and the resulting Mp,
#     and the metric-adjoint identity ‖B·Rxy_13 − Rxy_31ᵀ·B‖.
# B5: iel=first(cpl) may drop the ³D₁ doublet channel. Print per-channel Mp; see if dropped channels
#     carry the scattering phase.
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
nx=16; ny=40; xmax=28.0; ymax=60.0; nθ=12; alpha=1.0
θ_deg=10.0; θ=θ_deg*π/180
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0; μ=2m/3

α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)
V           = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
Rxy_13      = Rxy_13_matrix_optimized(α, grid)
Nx = matrices_optimized.compute_overlap_matrix(grid.nx, grid.xx)
Ny = matrices_optimized.compute_overlap_matrix(grid.ny, grid.yy)
Bm = kron(Matrix{Float64}(I,α.nchmax,α.nchmax), kron(Nx,Ny))
bE,bψ = bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
ψ_in = compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
ψ_sc,A,b = solve_scattering_equation(E_cm+E_d,α,grid,potname,ψ_in,θ_deg=θ_deg)
q=sqrt(2*μ*amu*E_cm)/ħ
ψ_total=ψ_in.+ψ_sc; Ψbar=ψ_total.+Rxy*ψ_total; VΨ=V*Ψbar

# c-norm
n2b=size(φ_d,2)
evec=ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
Ix=[ (i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)

# bras: ĥ⁻ and regular-F, built like compute_initial_state_vector
function build_bra(kind)
    nxg=grid.nx; nyg=grid.ny; nch=length(α.l)
    bra=zeros(ComplexF64, nxg*nyg*nch)
    cpl=Int[]; m2b=Int[]; λmax=-1
    for iα in 1:nch
        i2b=α.α2bindex[iα]
        if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
            l=α.α2b.l[i2b]; mc=(l==0 ? 1 : (l==2 ? 2 : 0))
            if mc>0; push!(cpl,iα); push!(m2b,mc); λmax=max(λmax,Int(round(α.λ[iα]))); end
        end
    end
    Hall=Vector{OffsetArray{ComplexF64,1}}(undef,nyg)
    for iy in 1:nyg
        z=ComplexF64(q*grid.yi[iy]*exp(im*θ))
        fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=λmax,mode=4)
        h = ifail!=0 ? zeros(ComplexF64,λmax+1) : (kind==:hminus ? gc.-im.*fc : fc)
        Hall[iy]=OffsetArray(h,0:λmax)
    end
    for (idx,iα) in enumerate(cpl)
        λ=Int(round(α.λ[iα])); mc=m2b[idx]
        for ix in 1:nxg, iy in 1:nyg
            i=(iα-1)*nxg*nyg+(ix-1)*nyg+iy
            fx=grid.ϕx[ix]; fy=grid.ϕy[iy]
            (abs(fx)<1e-15||abs(fy)<1e-15) && continue
            bra[i]=(φ_d[ix,mc]*Hall[iy][λ])/(fx*fy)
        end
    end
    return bra, cpl, m2b
end

nxg=grid.nx; nyg=grid.ny; blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
res2S(Mp)=(f=-(1.0/E_cm)/C_n*Mp; S=1+2im*q*f; (rad2deg(0.5*angle(S)),abs(S)))

println("="^80)
@printf("rel_res=%.1e q=%.4f  benchmark: δ=105.49°, η=0.4649\n", norm(A*ψ_sc-b)/norm(b), q)

# ---- B1: metric-adjoint identity + 4 rearrangement maps ----
println("\n########## B1: Rxy metric-adjoint audit ##########")
@printf("metric-adjoint identity ‖B·Rxy_13 − Rxy_31ᵀ·B‖/‖Rxy_31ᵀ·B‖ = %.3e  (0 ⇒ Rxy_13=B⁻¹Rxy_31ᵀB)\n",
        norm(Bm*Rxy_13 .- transpose(Rxy_31)*Bm)/norm(transpose(Rxy_31)*Bm))
@printf("symmetry of Rxy in metric: ‖B·Rxy − Rxyᵀ·B‖/‖Rxyᵀ·B‖ = %.3e\n",
        norm(Bm*Rxy .- transpose(Rxy)*Bm)/norm(transpose(Rxy)*Bm))
for kind in (:hminus,:Freg)
    bra,cpl,m2b=build_bra(kind); iel=first(cpl)
    println("\n--- bra=$kind ---")
    maps = Dict(
      "Rxy*bra"            => Rxy*bra,
      "2*Rxy_13*bra"       => 2 .*(Rxy_13*bra),
      "B\\(Rxyᵀ B bra)"     => Bm\(transpose(Rxy)*(Bm*bra)),
      "B\\(Rxy_31ᵀ B bra)"  => Bm\(transpose(Rxy_31)*(Bm*bra)),
    )
    for (nm,Rb) in maps
        Mp=transpose(blk(Rb,iel))*blk(VΨ,iel)
        δ,η=res2S(Mp)
        @printf("  %-20s |Mp|=%.3e  δ=%8.3f°  η=%.4f\n", nm, abs(Mp), δ, η)
    end
end

# ---- B5: per-channel Mp (is first(cpl) dropping signal?) ----
println("\n########## B5: per-channel projection (bra=hminus, Rbra=Rxy*bra) ##########")
bra,cpl,m2b=build_bra(:hminus); Rb=Rxy*bra
@printf("%-4s %-4s %-5s %-5s %-5s %-5s | %-11s %-9s\n","#","iα","λ","J3","J12","l2b","|Mp_chan|","arg(°)")
Mtot=0.0im
for (idx,iα) in enumerate(cpl)
    i2b=α.α2bindex[iα]
    Mc=transpose(blk(Rb,iα))*blk(VΨ,iα)
    global Mtot+=Mc
    @printf("%-4d %-4d %-5d %-5.1f %-5.1f %-5d | %-11.3e %-9.2f\n",
        idx,iα,Int(round(α.λ[iα])),α.J3[iα],α.α2b.J12[i2b],α.α2b.l[i2b],abs(Mc),rad2deg(angle(Mc)))
end
δ1,η1=res2S(transpose(blk(Rb,first(cpl)))*blk(VΨ,first(cpl)))
δs,ηs=res2S(Mtot)
@printf("\nfirst(cpl) only : δ=%8.3f° η=%.4f\n", δ1,η1)
@printf("sum all cpl     : δ=%8.3f° η=%.4f   (naive coherent sum, no Wigner recoupling)\n", δs,ηs)
println("="^80)
