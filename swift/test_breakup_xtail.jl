# test_breakup_xtail.jl — quick check (Jin 2026-06-30): does the breakup Faddeev component populate
# large x (beyond the deuteron range ~5 fm) under uniform CS, and more so at higher energy?
# Reuses the test_3body_greens.jl doublet setup (modules + bound2b + scaled V + Rxy + GMRES solve),
# then reshapes the scattered Faddeev component ψ_sc and scans |ψ_sc(x, y_fixed)| vs x.
# Touches no Rxy / no anisotropic rotation — purely reads the existing uniform-CS solution.
using LinearAlgebra, Printf
const BB="/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/general_modules/channels.jl"); using .channels
include(BB*"/general_modules/mesh.jl");     using .mesh
include(BB*"/swift/matrices_optimized.jl"); using .matrices_optimized
include(BB*"/swift/scattering.jl");          using .Scattering
include(BB*"/swift/twobody.jl");             using .twobodybound
include(BB*"/swift/coulcc.jl");              using .CoulCC
const ħ=197.3269718; const m=1.0079713395678829; const amu=931.49432
fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
nθ=12; alpha=1.0; potname="MT"; z1z2=0.0; μ_y=2m/3

function xtail(E_lab; nx=24, ny=120, xmax=30.0, ymax=120.0, θ_deg=10.0, lmx=0, λmx=2, j2bmx=1.0)
    θ=θ_deg*π/180; E_cm=(2/3)*E_lab
    α=α3b(fermion,Jtot,T,Parity,lmx,0,λmx,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,j2bmx)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    bE,bψ=bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    Efull=E_cm+E_d
    _,B,Tm,V,Rxy,Rxy_31,Tx_ch,Ty_ch,_,Nx,Ny,V_x_full =
        Scattering.compute_scattering_matrix(Efull,α,grid,potname,θ_deg=θ_deg,assemble_A=false)
    nch=length(α.l); nxg=grid.nx; nyg=grid.ny
    # entrance: deuteron-coupled λ=0 (³S₁) channel
    iα0=0
    for iα in 1:nch
        i2b=α.α2bindex[iα]
        if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1 &&
           α.α2b.l[i2b]==0 && Int(round(α.λ[iα]))==0
            iα0=iα; break
        end
    end
    ψin=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
    for iα in 1:nch; Int(round(α.λ[iα]))==0 || (ψin[(iα-1)*nxg*nyg+1:iα*nxg*nyg].=0); end
    bsrc=V*(Rxy*ψin)
    ψsc,_=gmres_scattering(Efull,B,Tm,V,Rxy,bsrc,α,grid,Tx_ch,Ty_ch,V_x_full,Nx,Ny; reltol=1e-9)
    # reshape entrance-channel block: index = (ix-1)*ny + iy  →  M[iy, ix]
    blk=ψsc[(iα0-1)*nxg*nyg+1:iα0*nxg*nyg]
    M=reshape(blk, nyg, nxg)               # M[iy, ix]
    # |ψ_sc(x)| summed over y (L2 over y) = x-profile of the scattered Faddeev component
    xprof=[sqrt(sum(abs2, @view M[:,ix])) for ix in 1:nxg]
    peak=maximum(xprof)
    @printf("\n=== E_lab=%.1f MeV  (E_cm=%.2f, deuteron breakup thr ≈2.22) θ=%.0f° ===\n",E_lab,E_cm,θ_deg)
    @printf("  x-profile ‖ψ_sc(x,·)‖_y of the entrance ³S₁ channel (peak=%.3e):\n",peak)
    @printf("   %8s %12s %10s\n","x(fm)","‖ψ_sc‖_y","/peak")
    for ix in 1:nxg
        x=grid.xi[ix]
        (x<2 || (2<=x<6) || (ix%3==0) || x>20) || continue
        @printf("   %8.2f %12.4e %10.2e\n",x,xprof[ix],xprof[ix]/peak)
    end
    # fraction of the x-norm beyond the deuteron range
    tot=sqrt(sum(abs2,xprof))
    for xc in (5.0,10.0,15.0)
        f=sqrt(sum(abs2, xprof[grid.xi.>xc]))/tot
        @printf("  ‖ψ_sc‖ fraction with x>%.0f fm : %.1f%%\n",xc,100f)
    end
end

xtail(14.1)
xtail(42.0)
