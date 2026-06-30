# test_3body_yspline_scatt.jl — Step 2a (Jin 2026-06-30): 3-body n-d scattering with y on the Hermite
# spline basis, UNIFORM CS (Jin's call: smooth-ECS 3-body Rxy parked, see devlog). This sub-step builds the
# scattering matrix (kinetic ×e^{-2iθ}, V back-rotated, real Rxy reused from the validated θ=0 bound-state
# assembly, B=kron(Nx,Sy_real)), the source Ω=φ_d(x)·F_λ(qy·e^{iθ}) with the y-part PROJECTED onto the spline
# basis (Sy\∫φF_λ), solves A·ψ_sc = 2·V·Rxy·Ω, and checks ψ_sc solves + carries breakup flux. The Green's
# amplitude (δ,η) is the NEXT sub-step (2b); here we only verify the scattering plumbing on the spline-y.

using LinearAlgebra, Printf, FastGaussQuadrature, OffsetArrays
const BB = "/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/general_modules/channels.jl"); using .channels
include(BB*"/general_modules/mesh.jl");     using .mesh
include(BB*"/swift/matrices_optimized.jl"); using .matrices_optimized
include(BB*"/swift/laguerre.jl");           using .Laguerre
include(BB*"/swift/Gcoefficient.jl");       using .Gcoefficient
include(BB*"/swift/splines.jl");            using .Splines
include(BB*"/swift/twobody.jl");            using .twobodybound
include(BB*"/swift/coulcc.jl");             using .CoulCC
include(BB*"/swift/scattering.jl");         import .Scattering
import IterativeSolvers

const ħ = 197.3269718; const m = 1.0079713395678829; const amu = 931.49432
const ħ2_2μy = ħ^2 * 0.75 / (m * amu)

function spline_y_ops(ymesh::SplineMesh, λset; nq=8)
    nd = ymesh.ndof
    Sy = zeros(Float64, nd, nd)
    Ky = Dict{Int,Matrix{Float64}}(λ => zeros(Float64, nd, nd) for λ in λset)
    uq, wq = gausslegendre(nq)
    for ip in 1:ymesh.nint
        a = ymesh.knots[ip]; h = ymesh.knots[ip+1]-ymesh.knots[ip]
        for g in 1:nq
            y = a + 0.5h*(uq[g]+1.0); w = 0.5h*wq[g]
            idx, φ, φ1, φ2 = spline_functions(ymesh, y)
            for (li,i) in enumerate(idx), (lj,j) in enumerate(idx)
                Sy[i,j] += w*φ[li]*φ[lj]
                for λ in λset
                    Ky[λ][i,j] += w*φ[li]*(ħ2_2μy*(-φ2[lj] + λ*(λ+1)/y^2*φ[lj]))
                end
            end
        end
    end
    return Sy, Ky
end

function scatt(; E_lab=14.1, nx=20, xmax=18.0, ymax=60.0, nyint=80, ncol=3, nq=8, alpha=1.0, θ_deg=10.0)
    θ = θ_deg*π/180; E_cm = (2/3)*E_lab; z1z2 = 0.0; μ_y = 2m/3
    fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
    α = α3b(fermion,Jtot,T,Parity,0,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0); nch = α.nchmax
    grid = initialmesh(12, nx, 4, xmax, ymax, alpha)
    # x-side Lagrange operators WITH uniform CS (Tx_ch carries e^{-2iθ}); V back-rotated
    _, Tx_ch, _, Nx, _ = T_matrix_optimized(α, grid, return_components=true, θ_deg=θ_deg)
    _, _, V_x_full = V_matrix_optimized_scaled(α, grid, "MT"; θ_deg=θ_deg, return_components=true, return_vsector_blocks=true)
    # deuteron (x) at θ, C_n-normalized (removes the bound2b eigenvector gauge phase that the bilinear
    # amplitude ∝ φ_d² would otherwise carry; C_n = φ_dᵀ B_x φ_d · e^{iθ})
    bE, bψ = bound2b(grid, "MT", θ_deg=θ_deg); φ_d = ComplexF64.(bψ[1]); E_d = real(bE[1])
    n2b = size(φ_d,2)
    cnorm(φ, θl) = begin
        ev = ComplexF64[φ[j,ich]/grid.ϕx[j] for ich in 1:n2b for j in 1:nx]
        (transpose(ev)*kron(Matrix{ComplexF64}(I,n2b,n2b), ComplexF64.(Nx))*ev)*exp(im*θl)
    end
    φ_d = φ_d ./ sqrt(cnorm(φ_d, θ))
    # θ=0 deuteron for the Born term, separately C_n-normalized
    bE0, bψ0 = bound2b(grid, "MT", θ_deg=0.0); φ_d0 = ComplexF64.(bψ0[1])
    φ_d0 = φ_d0 ./ sqrt(cnorm(φ_d0, 0.0))
    Efull = E_cm + E_d
    # y-side spline (real Sy, Ky; CS via ×e^{-2iθ} on the kinetic, uniform-CS scheme)
    ymesh = init_spline_mesh([(0.0, ymax, nyint, 1.0)]; ncol=ncol)
    λset = sort(unique(Int.(round.(α.λ))))
    Sy, Ky = spline_y_ops(ymesh, λset; nq=nq)
    nd = ymesh.ndof
    dropped = [1, ymesh.nint*ncol+1]; ncol==3 && push!(dropped, ymesh.nint*ncol+2)
    keep = setdiff(1:nd, dropped); ndy = length(keep)
    pos = fill(0, nd); for (k,d) in enumerate(keep); pos[d]=k; end
    Syk = Sy[keep,keep]; Kyk = Dict(λ => Ky[λ][keep,keep] for λ in λset)
    e2iθ = exp(-2im*θ)

    # ---- mixed Rxy (REAL, validated; uniform CS keeps it real) ----
    uq, wq = gausslegendre(nq); nyq = ymesh.nint*nq
    yq = Float64[]; wyq = Float64[]
    for ip in 1:ymesh.nint
        a = ymesh.knots[ip]; h = ymesh.knots[ip+1]-ymesh.knots[ip]
        for g in 1:nq; push!(yq, a+0.5h*(uq[g]+1.0)); push!(wyq, 0.5h*wq[g]); end
    end
    out_idx = [spline_functions(ymesh, yq[j])[1] for j in 1:nyq]
    out_val = [Float64.(real.(spline_functions(ymesh, yq[j])[2])) for j in 1:nyq]
    λmax = maximum(α.λ); lmax = maximum(α.l)
    Yλo, Yli, Yλi = Gcoefficient.initialY(λmax, lmax, grid.nθ, nx, nyq, grid.cosθi, grid.xi, yq)
    Gq = Gcoefficient.Gαα(grid.nθ, nyq, nx, α, Yλo, Yli, Yλi, grid)
    a31,b31,c31,d31 = -0.5,1.0,-0.75,-0.5
    Ntot = nch*nx*ndy
    Rxy = zeros(Float64, Ntot, Ntot)
    oidx(iα,ix,ky) = (iα-1)*nx*ndy + (ix-1)*ndy + ky
    for ix in 1:nx
        xa = grid.xi[ix]; invϕx = 1.0/grid.ϕx[ix]
        for iyq in 1:nyq
            ya = yq[iyq]
            for iθ in 1:grid.nθ
                cs = grid.cosθi[iθ]; dcs = grid.dcosθi[iθ]
                πb = sqrt(a31^2*xa^2+b31^2*ya^2+2*a31*b31*xa*ya*cs)
                ξb = sqrt(c31^2*xa^2+d31^2*ya^2+2*c31*d31*xa*ya*cs); ξb = min(ξb, ymax-1e-9)
                fπb = real.(lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx))
                idξ, Sξ4, _, _ = spline_functions(ymesh, ξb); Sξ = Float64.(real.(Sξ4))
                base = dcs*xa*ya/(πb*ξb)*invϕx
                for iα in 1:nch, iαp in 1:nch
                    G = Gq[iθ,iyq,ix,iα,iαp,1]; abs(G)<1e-14 && continue
                    coef = 2.0*base*G
                    for (lo,gdo) in enumerate(out_idx[iyq])
                        ko = pos[gdo]; ko==0 && continue
                        wpo = wyq[iyq]*out_val[iyq][lo]; wpo==0.0 && continue
                        for (li,gdi) in enumerate(idξ)
                            ki = pos[gdi]; ki==0 && continue
                            v = coef*wpo*Sξ[li]; v==0.0 && continue
                            for ixp in 1:nx
                                Rxy[oidx(iα,ix,ko), oidx(iαp,ixp,ki)] += v*fπb[ixp]
                            end
                        end
                    end
                end
            end
        end
    end

    # ---- MATRIX-FREE operators (no dense A; reshape-based kron matvecs + one Rc matvec) ----
    # Channel block index = (ix-1)*ndy + ky, ky innermost → kron(A_x,A_y)·vec(M[ky,ix]) = vec(A_y·M·A_xᵀ).
    blk(iα) = (iα-1)*nx*ndy
    rsh(v,iα) = reshape(view(v, blk(iα)+1:blk(iα)+nx*ndy), ndy, nx)
    Nxc = ComplexF64.(Nx); Sykc = ComplexF64.(Syk)
    Txc = [ComplexF64.(Tx_ch[iα]) for iα in 1:nch]          # already carries e^{-2iθ}
    Kcs = Dict(λ => e2iθ .* ComplexF64.(Kyk[λ]) for λ in keys(Kyk))   # y-kinetic ×e^{-2iθ}
    Vx  = [ComplexF64.(V_x_full[iα,iαp]) for iα in 1:nch, iαp in 1:nch]
    Rc  = ComplexF64.(Rxy)
    λof = [Int(round(α.λ[iα])) for iα in 1:nch]
    nzV = [!all(==(0.0+0im), Vx[iα,iαp]) for iα in 1:nch, iαp in 1:nch]
    # V·Rxy applied with a given set of x-V blocks (Sy cancels: kron(Vx,I)·Rc)
    function applyVR(x, Vb)
        w = Rc*x; out = zeros(ComplexF64, Ntot)
        for iα in 1:nch
            Mo = zeros(ComplexF64, ndy, nx)
            for iαp in 1:nch
                nzV[iα,iαp] || continue
                Mo .+= rsh(w,iαp) * transpose(Vb[iα,iαp])
            end
            out[blk(iα)+1:blk(iα)+nx*ndy] .= vec(Mo)
        end
        out
    end
    function applyA(x)
        out = zeros(ComplexF64, Ntot)
        for iα in 1:nch
            M = rsh(x,iα); λ = λof[iα]
            Mo = Efull*(Sykc*M*transpose(Nxc)) .- (Sykc*M*transpose(Txc[iα]) .+ Kcs[λ]*M*transpose(Nxc))
            for iαp in 1:nch
                nzV[iα,iαp] || continue
                Mo .-= Sykc*rsh(x,iαp)*transpose(Vx[iα,iαp])      # − V
            end
            out[blk(iα)+1:blk(iα)+nx*ndy] .= vec(Mo)
        end
        out .-= applyVR(x, Vx)                                     # − V·Rxy
        out
    end
    # within-channel block preconditioner P[iα] = Efull·kron(Nx,Sy) − kron(Tx,Sy) − kron(Nx,Ky_cs) − kron(Vx_ii,Sy)
    Pf = Vector{Any}(undef, nch)
    for iα in 1:nch
        λ = λof[iα]
        P = Efull*kron(Nxc,Sykc) .- kron(Txc[iα],Sykc) .- kron(Nxc,Kcs[λ]) .- kron(Vx[iα,iα],Sykc)
        Pf[iα] = lu(P)
    end
    function applyPinv(x)
        out = similar(x, ComplexF64)
        for iα in 1:nch
            out[blk(iα)+1:blk(iα)+nx*ndy] .= Pf[iα] \ x[blk(iα)+1:blk(iα)+nx*ndy]
        end
        out
    end

    # ---- source Ω = φ_d(x)·F_λ(qy·e^{iθ}), y projected onto spline ----
    q = sqrt(2*μ_y*amu*E_cm)/ħ
    if CoulCC.libcoulcc[] == C_NULL; CoulCC.load_library(); end
    # spline projection of F_λ(q·y·e^{iθ}) for each λ: c = Sy \ p, p[i]=∫φ_i F_λ dy
    Fcoef = Dict{Int,Vector{ComplexF64}}()
    for λ in λset
        p = zeros(ComplexF64, nd)
        for ip in 1:ymesh.nint
            a = ymesh.knots[ip]; h = ymesh.knots[ip+1]-ymesh.knots[ip]
            for g in 1:nq
                y = a+0.5h*(uq[g]+1.0); w = 0.5h*wq[g]
                z = ComplexF64(q*y*exp(im*θ))
                fc,_,_,_,_,_ = coulcc(z, ComplexF64(0.0), 0; lmax=Int(λmax), mode=4)
                idx, φ, _, _ = spline_functions(ymesh, y)
                for (li,i) in enumerate(idx); p[i] += w*real(φ[li])*fc[λ+1]; end
            end
        end
        c = Sy \ p
        Fcoef[λ] = c[keep]
    end
    # entrance = doublet λ=0 deuteron channel(s); source masked to λ=0 (incoming n-d S-wave)
    function build_Omega(Fc, φdd)
        Ω = zeros(ComplexF64, Ntot)
        for iα in 1:nch
            i2b = α.α2bindex[iα]
            (Int(round(α.α2b.J12[i2b]))==1 && Int(round(α.α2b.s12[i2b]))==1) || continue
            Int(round(α.λ[iα]))==0 || continue          # ENTRANCE: spectator λ=0 only
            l2 = α.α2b.l[i2b]; mc = (l2==0 ? 1 : (l2==2 ? 2 : 0)); mc>0 || continue
            cy = Fc[0]
            for ix in 1:nx
                xcoef = φdd[ix,mc]/grid.ϕx[ix]
                for ky in 1:ndy; Ω[oidx(iα,ix,ky)] = xcoef*cy[ky]; end
            end
        end
        return Ω
    end
    Ω = build_Omega(Fcoef, φ_d)
    bsrc = applyVR(Ω, Vx)                     # 2·V·Rxy·Ω (Rxy already ×2)
    Aop = Scattering.MatVecOperator{ComplexF64}(applyA, Ntot)
    Pop = Scattering.PreconditionerOperator{ComplexF64}(applyPinv, Ntot)
    ψsc, hist = IterativeSolvers.gmres(Aop, bsrc; Pl=Pop, reltol=1e-9, maxiter=600, log=true)
    resid = norm(applyA(ψsc) - bsrc)/norm(bsrc)

    # ---- Green's amplitude (port of test_3body_greens): f=−(flux_norm·e^{iθ}/E_cm)(f_sc+f_brn) ----
    entrance = [iα for iα in 1:nch if Int(round(α.λ[iα]))==0 &&
                Int(round(α.α2b.J12[α.α2bindex[iα]]))==1 && Int(round(α.α2b.s12[α.α2bindex[iα]]))==1]
    projb(bra,ket) = sum(iα -> sum(bra[blk(iα)+1:blk(iα)+nx*ndy] .* ket[blk(iα)+1:blk(iα)+nx*ndy]), entrance)
    jac = exp(im*2.0*θ)
    f_sc = jac * projb(Ω, applyVR(ψsc, Vx))
    # Born term on the REAL axis (θ=0): θ=0 V blocks, real F_λ source, same real Rxy
    Vx0b = V_x_pair_blocks(α, grid, "MT")
    Vx0 = [ComplexF64.(Vx0b[iα,iαp]) for iα in 1:nch, iαp in 1:nch]
    Fcoef0 = Dict{Int,Vector{ComplexF64}}()
    for λ in λset
        p = zeros(ComplexF64, nd)
        for ip in 1:ymesh.nint
            a = ymesh.knots[ip]; h = ymesh.knots[ip+1]-ymesh.knots[ip]
            for g in 1:nq
                y = a+0.5h*(uq[g]+1.0); w = 0.5h*wq[g]; z = ComplexF64(q*y)
                fc,_,_,_,_,_ = coulcc(z, ComplexF64(0.0), 0; lmax=Int(λmax), mode=4)
                idx, φ, _, _ = spline_functions(ymesh, y)
                for (li,i) in enumerate(idx); p[i] += w*real(φ[li])*fc[λ+1]; end
            end
        end
        Fcoef0[λ] = (Sy \ p)[keep]
    end
    Ω0 = build_Omega(Fcoef0, φ_d0)
    f_brn = projb(Ω0, applyVR(Ω0, Vx0))
    flux_norm = (μ_y/(m/2))^(1/4)
    f = -(flux_norm*exp(im*θ)/E_cm)*(f_sc + f_brn)
    S = 1 + 2im*q*f
    δ = rad2deg(0.5*angle(S)); δ = δ<0 ? δ+180 : δ; η = abs(S)
    # diagnostics
    @printf("E_lab=%.1f θ=%.0f° nx=%d nyint=%d ymax=%.0f xmax=%.0f q=%.4f E_d=%.4f gmres=%d resid=%.2e\n",
            E_lab, θ_deg, nx, nyint, ymax, xmax, q, E_d, hist.iters, resid)
    tot = 0.0; for iα in 1:nch; tot += norm(ψsc[blk(iα)+1:blk(iα)+nx*ndy])^2; end; tot=sqrt(tot)
    @printf("  ‖Ω‖=%.3e ‖bsrc‖=%.3e ‖ψ_sc‖=%.3e\n", norm(Ω), norm(bsrc), tot)
    @printf("  f_sc=%.3f%+.3fi  f_brn=%.3f%+.3fi  ⇒  δ=%8.3f°  η=%.4f\n",
            real(f_sc),imag(f_sc), real(f_brn),imag(f_brn), δ, η)
    @printf("  [doublet-14.1 benchmark: δ=105.49°, η=0.4649]\n")
    return δ, η
end

println("="^70)
println(" Step 2a: 3-body n-d scattering, spline-y, uniform CS — solve + flux check")
println("="^70)
# Lit check (2011 PRC Table III): 3-body needs r_max≈100 fm BOTH directions; θ∈[4°,12.5°] window @14.1.
# My earlier xmax≤36 was far too small → δ slid. Push BOTH boxes toward ~100 fm (balanced), θ=6°, look for
# a δ PLATEAU at the benchmark δ=105.50, η=0.4653.
# GMRES validated (≡ dense δ=97.09/η=0.60, 15 iters). Now balanced-box convergence toward r_max~100 fm,
# θ=6° (window θ_max=14.2°@14.1). Watch δ→105.50, η→0.4653 PLATEAU. (Big r_max~100 belongs on heliumx.)
println(">>> balanced-box convergence, θ=6°, target δ=105.50/η=0.4653")
scatt(E_lab=14.1, nx=30, xmax=50.0, ymax=50.0, nyint=34, nq=6, θ_deg=6.0)
scatt(E_lab=14.1, nx=40, xmax=70.0, ymax=70.0, nyint=44, nq=6, θ_deg=6.0)
scatt(E_lab=14.1, nx=46, xmax=90.0, ymax=90.0, nyint=50, nq=6, θ_deg=6.0)
