# test_eq17_mc.jl — RIGOROUS multichannel n-d doublet S-matrix (idea-pk path a).
# Jtot=½⁺ has TWO channel-spin states: (λ=0,𝕊=½)=doublet ²S and (λ=2,𝕊=3/2).
# Solve the Faddeev scattering eq SEPARATELY for each incident λ (mask ψ_in to that λ, deuteron
# carries its full S+D in x), build the 2×2 collision matrix U[out,in], diagonalize S → the two
# eigenphases (η_j e^{2iδ_j}); the doublet = the λ=0-dominant eigenvalue.
# Benchmark (Tab.III): doublet δ=105.49°, η=0.4649.
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

# Riccati ĥ⁻_λ(qy e^{iθ}) bra for a SINGLE λ_out (deuteron S+D in x, only λ=λ_out channels)
function bra_lambda(α,grid,φ_d,q,λ_out)
    nxg=grid.nx; nyg=grid.ny; bra=zeros(ComplexF64,nxg*nyg*length(α.l))
    Hall=Vector{ComplexF64}(undef,nyg)
    for iy in 1:nyg
        z=ComplexF64(q*grid.yi[iy]*exp(im*θ)); fc,gc,_,_,_,ifail=coulcc(z,ComplexF64(0.0),λ_out;lmax=λ_out,mode=4)
        Hall[iy]= ifail!=0 ? 0.0im : (gc[end]-im*fc[end])   # ĥ⁻_{λ_out}
    end
    for iα in 1:length(α.l)
        Int(round(α.λ[iα]))==λ_out || continue
        i2b=α.α2bindex[iα]
        (Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1) || continue
        l=α.α2b.l[i2b]; mc=(l==0 ? 1 : (l==2 ? 2 : 0)); mc>0 || continue
        for ix in 1:nxg, iy in 1:nyg
            i=(iα-1)*nxg*nyg+(ix-1)*nyg+iy; fx=grid.ϕx[ix]; fy=grid.ϕy[iy]
            (abs(fx)<1e-15||abs(fy)<1e-15) && continue
            bra[i]=(φ_d[ix,mc]*Hall[iy])/(fx*fy)
        end
    end
    return bra
end

function run(nx,ny,xmax,ymax)
    α=α3b(fermion,Jtot,T,Parity,2,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0)
    grid=initialmesh(nθ,nx,ny,xmax,ymax,alpha)
    Rxy,Rxy_31=Rxy_matrix_optimized(α,grid); V=V_matrix_optimized_scaled(α,grid,potname,θ_deg=θ_deg)
    bE,bψ=bound2b(grid,potname,θ_deg=θ_deg); φ_d=ComplexF64.(bψ[1]); E_d=real(bE[1])
    q=sqrt(2*μ*amu*E_cm)/ħ
    n2b=size(φ_d,2); evec=ComplexF64[φ_d[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:grid.nx]
    Ix=[(i==j ? 1.0 : 0.0)+(-1.0)^(j-i)/sqrt(grid.xx[i]*grid.xx[j]) for i in 1:grid.nx,j in 1:grid.nx]
    C_n=(transpose(evec)*kron(Matrix{Float64}(I,n2b,n2b),Ix)*evec)*exp(im*θ)
    nxg=grid.nx; nyg=grid.ny; blk(v,iα)=v[(iα-1)*nxg*nyg+1:iα*nxg*nyg]
    λs=[0,2]
    # per-incident-λ solve → store the Faddeev component ψ_total^(in)
    VRxyψ=Dict{Int,Vector{ComplexF64}}()
    for λin in λs
        ψ_in=compute_initial_state_vector(grid,α,φ_d,E_cm,z1z2,θ=θ)
        for iα in 1:length(α.l)   # mask to λ=λin
            Int(round(α.λ[iα]))==λin || (ψ_in[(iα-1)*nxg*nyg+1:iα*nxg*nyg].=0)
        end
        ψ_sc,_,_=solve_scattering_equation(E_cm+E_d,α,grid,potname,ψ_in,θ_deg=θ_deg)
        ψ_tot=ψ_in.+ψ_sc
        VRxyψ[λin]=V*(Rxy*ψ_tot)          # operator V·Rxy on the Faddeev component (source-mirror)
    end
    bras=Dict(λ=>bra_lambda(α,grid,φ_d,q,λ) for λ in λs)
    # 2×2 collision matrix, scan Jacobian n
    println("  doublet target: δ=105.49°, η=0.4649")
    for n in 0.0:0.5:4.0
        pref=-(1.0/E_cm)/C_n*exp(im*n*θ)
        f=zeros(ComplexF64,2,2)
        for (io,λo) in enumerate(λs), (ii,λi) in enumerate(λs)
            Mel=0.0im
            for iα in 1:length(α.l)
                Int(round(α.λ[iα]))==λo || continue
                Mel+=transpose(blk(bras[λo],iα))*blk(VRxyψ[λi],iα)
            end
            f[io,ii]=pref*Mel
        end
        S=I(2)+2im*q*f
        evals,evecs=eigen(S)
        # doublet = eigenvalue whose eigenvector is λ=0-dominant (component 1)
        id=argmax(abs.(evecs[1,:]))
        sd=evals[id]
        @printf("  e^{%.1fθ}: doublet δ=%8.3f° η=%.4f | other δ=%8.3f° η=%.4f | offdiag|f12|=%.2e\n",
                n, rad2deg(0.5*angle(sd)),abs(sd),
                rad2deg(0.5*angle(evals[3-id])),abs(evals[3-id]), abs(f[1,2]))
    end
end
println("===== nx=16 ny=40 ====="); run(16,40,28.0,60.0)
println("===== nx=20 ny=50 ====="); run(20,50,32.0,70.0)
