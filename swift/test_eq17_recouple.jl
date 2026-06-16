# test_eq17_recouple.jl — full Blatt-Biedenharn channel-spin recoupling of the Eq.17 amplitude
# over ALL deuteron channels (B5 fix), with the 2-body-anchored prefactor −(1/E_cm)/C_n (A5).
# Reads the doublet ²S_{1/2} eigenphase from the (J=½,π=+1) block's (λ=0,𝕊=0.5) diagonal.
# Benchmark (Tab.III doublet 14.1): Re δ=105.49°, η=0.4649.
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
E_lab=14.1; E_cm=(2/3)*E_lab; potname="MT"; z1z2=0.0; μ=2m/3

function setup(nx,ny,xmax,ymax)
    α   = α3b(fermion,Jtot,T,Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)
    grid= initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    V           = V_matrix_optimized_scaled(α, grid, potname, θ_deg=θ_deg)
    Rxy, Rxy_31 = Rxy_matrix_optimized(α, grid)
    Nx=matrices_optimized.compute_overlap_matrix(grid.nx,grid.xx)
    Ny=matrices_optimized.compute_overlap_matrix(grid.ny,grid.yy)
    Bm=kron(Matrix{Float64}(I,α.nchmax,α.nchmax),kron(Nx,Ny))
    bE,bψ=bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    ψ_in=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
    ψ_sc,A,b=solve_scattering_equation(E_cm+E_d,α,grid,potname,ψ_in,θ_deg=θ_deg)
    q=sqrt(2*μ*amu*E_cm)/ħ
    ψ_total=ψ_in.+ψ_sc; Ψbar=ψ_total.+Rxy*ψ_total; VΨ=V*Ψbar
    n2b=size(φ_d,2)
    evec=ComplexF64[ φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx ]
    Ix=[ (i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx, j in 1:grid.nx ]
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)
    return (;α,grid,V,Rxy,Rxy_31,Bm,φ_d,q,VΨ,C_n,relres=norm(A*ψ_sc-b)/norm(b))
end

function bra_vec(s,kind)
    α=s.α; grid=s.grid; nxg=grid.nx; nyg=grid.ny; nch=length(α.l)
    bra=zeros(ComplexF64,nxg*nyg*nch); cpl=Int[]; m2b=Int[]; λmax=-1
    for iα in 1:nch
        i2b=α.α2bindex[iα]
        if Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1
            l=α.α2b.l[i2b]; mc=(l==0 ? 1 : (l==2 ? 2 : 0))
            if mc>0; push!(cpl,iα); push!(m2b,mc); λmax=max(λmax,Int(round(α.λ[iα]))); end
        end
    end
    Hall=Vector{OffsetArray{ComplexF64,1}}(undef,nyg)
    for iy in 1:nyg
        z=ComplexF64(s.q*grid.yi[iy]*exp(im*θ))
        fc,gc,fcp,gcp,sig,ifail=coulcc(z,ComplexF64(0.0),0;lmax=λmax,mode=4)
        h=ifail!=0 ? zeros(ComplexF64,λmax+1) : (kind==:hminus ? gc.-im.*fc : fc)
        Hall[iy]=OffsetArray(h,0:λmax)
    end
    for (idx,iα) in enumerate(cpl)
        λ=Int(round(α.λ[iα])); mc=m2b[idx]
        for ix in 1:nxg, iy in 1:nyg
            i=(iα-1)*nxg*nyg+(ix-1)*nyg+iy; fx=grid.ϕx[ix]; fy=grid.ϕy[iy]
            (abs(fx)<1e-15||abs(fy)<1e-15) && continue
            bra[i]=(s.φ_d[ix,mc]*Hall[iy][λ])/(fx*fy)
        end
    end
    return bra,cpl
end

function run(nx,ny,xmax,ymax)
    s=setup(nx,ny,xmax,ymax); α=s.α; grid=s.grid
    nxg=grid.nx; nyg=grid.ny; blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    @printf("\n========== nx=%d ny=%d (rel_res=%.1e, q=%.4f) ==========\n",nx,ny,s.relres,s.q)
    for kind in (:hminus,:Freg)
        bra,cpl=bra_vec(s,kind); nd=length(cpl)
        for (adjname,Rb) in (("Rxy*bra",s.Rxy*bra),
                             ("B\\(Rxy_31ᵀ B bra)", s.Bm\(transpose(s.Rxy_31)*(s.Bm*bra))))
            M=zeros(ComplexF64,nd,nd)
            for io in 1:nd, ii in 1:nd
                M[io,ii]=transpose(blk(Rb,cpl[io]))*blk(s.VΨ,cpl[ii])
            end
            pref0=-(1.0/E_cm)/s.C_n
            best=(1e9,0.0,0.0,0.0)
            for n in 0.0:0.5:6.0
                f=(pref0*exp(im*n*θ)).*M
                U=compute_collision_matrix(f,s.q)
                Ucs,lab=Scattering.recouple_to_channel_spin(U,α,cpl)
                key=(Jtot,1); haskey(Ucs,key)||continue
                L=lab[key]; idx=findfirst(==("λ=0, 𝕊=0.5"),L); idx===nothing&&continue
                ss=Ucs[key][idx,idx]; δ=rad2deg(0.5*angle(ss)); η=abs(ss)
                d=abs(δ-105.49)+50*abs(η-0.4649)
                d<best[1] && (best=(d,n,δ,η))
            end
            @printf("  %-8s %-19s : best e^{%.1fθ}  δ=%8.3f°  η=%.4f  (target 105.49°,0.4649)\n",
                    kind,adjname,best[2],best[3],best[4])
        end
    end
end

println("benchmark doublet 14.1: Re δ=105.49°, η=0.4649")
run(16,40,28.0,60.0)
run(20,50,32.0,70.0)
