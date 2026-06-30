# test_3body_yspline_bound.jl — Step 2 plumbing test (Jin 2026-06-30): the 3-body MT bound state with
# the y coordinate on the HERMITE SPLINE basis (Galerkin, q-operator contour layer), x kept on
# Lagrange-Laguerre, θ=0 (real, contour trivial). Target = the all-Lagrange reference E(MT 3-body)
# ≈ −8.52 MeV (box-converged: −8.509 nx=ny=16 → −8.516 (18) → −8.520 (20)). Reproducing it validates
# the mixed-basis kron assembly + the new overlap B=kron(Nx,Sy) + the MIXED Rxy (Lagrange-x output at
# mesh points, spline-y output by Galerkin projection) — before any complex scaling.
#
# Consistency (derived 2026-06-30, no Sy⁻¹ needed): build Rxy in PROJECTION form (output y via
# Σ_q w_q φ_{i'}(y_q), replacing the Lagrange 1/ϕy mesh-value), then V·Rxy = kron(Vx, I_yDOF)·Rxy_proj
# because the standalone V=kron(Vx,Sy) and the coefficient-form Sy⁻¹ cancel in y. H=T+V+VRxy, B=kron(Nx,Sy).

using LinearAlgebra, Printf, FastGaussQuadrature
const BB = "/Users/jinlei/Desktop/code/swift.jl"
include(BB*"/general_modules/channels.jl"); using .channels
include(BB*"/general_modules/mesh.jl");     using .mesh
include(BB*"/swift/matrices_optimized.jl"); using .matrices_optimized
include(BB*"/swift/laguerre.jl");           using .Laguerre
include(BB*"/swift/Gcoefficient.jl");       using .Gcoefficient
include(BB*"/swift/splines.jl");            using .Splines
include(BB*"/swift/ecs.jl");                using .ECS

const ħ = 197.3269718; const m = 1.0079713395678829; const amu = 931.49432
const ħ2_2μy = ħ^2 * 0.75 / (m * amu)        # y reduced mass μ_y = 2m/3 → ħ²/2μ_y = 0.75 ħ²/m

# ---- spline y Galerkin operators (q-operator, θ=0 so contour trivial) ----
# Sy[i,j]=∫φ_iφ_j dy ; Ky_λ[i,j]=(ħ²/2μ_y)∫φ_i[ -φ_j'' + λ(λ+1)/y² φ_j ] dy   (q=1,q'=0)
function spline_y_ops(ymesh::SplineMesh, λset; nq=8)
    nd = ymesh.ndof
    Sy = zeros(Float64, nd, nd)
    Ky = Dict{Int,Matrix{Float64}}(λ => zeros(Float64, nd, nd) for λ in λset)
    uq, wq = gausslegendre(nq)
    for ip in 1:ymesh.nint
        a = ymesh.knots[ip]; h = ymesh.knots[ip+1] - ymesh.knots[ip]
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

function run_bound(; nx=20, xmax=18.0, ymax=18.0, nyint=34, ncol=3, nq=8, alpha=1.0)
    fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
    α = α3b(fermion,Jtot,T,Parity,0,0,2,0,0.5,0.5,0.5,0.5,0.5,0.5,MT,1.0)
    nch = α.nchmax
    grid = initialmesh(12, nx, 4, xmax, ymax, alpha)   # ny dummy (=4); y handled by spline
    # x-side Lagrange operators (reuse existing builders, θ=0)
    _, Tx_ch, _, Nx, _ = T_matrix_optimized(α, grid, return_components=true, θ_deg=0.0)
    Vx = V_x_pair_blocks(α, grid, "MT")                # per-pair x blocks [iα,iαp] (nx×nx)

    # y-side spline
    ymesh = init_spline_mesh([(0.0, ymax, nyint, 1.0)]; ncol=ncol)
    Sy, Ky = spline_y_ops(ymesh, sort(unique(Int.(round.(α.λ)))); nq=nq)
    nd = ymesh.ndof
    dropped = [1, ymesh.nint*ncol+1]; ncol==3 && push!(dropped, ymesh.nint*ncol+2)
    keep = setdiff(1:nd, dropped); ndy = length(keep)
    pos = fill(0, nd); for (k,d) in enumerate(keep); pos[d]=k; end   # global DOF → kept index (0 if dropped)
    Syk = Sy[keep,keep]
    Kyk = Dict(λ => Ky[λ][keep,keep] for λ in keys(Ky))

    # ---- y quadrature points for the mixed Rxy + G recomputation there ----
    uq, wq = gausslegendre(nq)
    nyq = ymesh.nint*nq
    yq = Vector{Float64}(undef, nyq); wyq = Vector{Float64}(undef, nyq)
    iq = 0
    for ip in 1:ymesh.nint
        a = ymesh.knots[ip]; h = ymesh.knots[ip+1]-ymesh.knots[ip]
        for g in 1:nq
            iq += 1; yq[iq] = a + 0.5h*(uq[g]+1.0); wyq[iq] = 0.5h*wq[g]
        end
    end
    # spline test-function values at the y-quad points (output projection Σ w φ_{i'}(yq))
    out_idx = Vector{Vector{Int}}(undef, nyq); out_val = Vector{Vector{Float64}}(undef, nyq)
    for j in 1:nyq
        id, S, _, _ = spline_functions(ymesh, yq[j]); out_idx[j]=id; out_val[j]=Float64.(real.(S))
    end

    # G coefficients recomputed AT the y-quad points (pseudo-grid: yi = yq, ny = nyq)
    λmax = maximum(α.λ); lmax = maximum(α.l)
    Yλout, Ylin, Yλin = Gcoefficient.initialY(λmax, lmax, grid.nθ, nx, nyq, grid.cosθi, grid.xi, yq)
    Gq = Gcoefficient.Gαα(grid.nθ, nyq, nx, α, Yλout, Ylin, Yλin, grid)   # [iθ,iyq,ix,iα,iαp,perm]

    # ---- mixed Rxy_proj (= 2·Rxy31_proj) ----
    a31,b31,c31,d31 = -0.5, 1.0, -0.75, -0.5
    Ntot = nch*nx*ndy
    Rxy = zeros(Float64, Ntot, Ntot)
    oidx(iα,ix,ky) = (iα-1)*nx*ndy + (ix-1)*ndy + ky
    for ix in 1:nx
        xa = grid.xi[ix]; invϕx = 1.0/grid.ϕx[ix]
        for iyq in 1:nyq
            ya = yq[iyq]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]; dcosθ = grid.dcosθi[iθ]
                πb = sqrt(a31^2*xa^2 + b31^2*ya^2 + 2*a31*b31*xa*ya*cosθ)
                ξb = sqrt(c31^2*xa^2 + d31^2*ya^2 + 2*c31*d31*xa*ya*cosθ)
                ξb = min(ξb, ymax - 1e-9)
                fπb = real.(lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx))
                idξ, Sξ, _, _ = spline_functions(ymesh, ξb); Sξ = Float64.(real.(Sξ))
                base = dcosθ*xa*ya/(πb*ξb)*invϕx
                for iα in 1:nch, iαp in 1:nch
                    G = Gq[iθ,iyq,ix,iα,iαp,1]
                    abs(G) < 1e-14 && continue
                    coef = 2.0 * base * G                      # ×2 Faddeev (Rxy=2Rxy31)
                    # output y projection × input y spline at ξb
                    for (lo, gdo) in enumerate(out_idx[iyq])
                        ko = pos[gdo]; ko == 0 && continue
                        wpo = wyq[iyq]*out_val[iyq][lo]
                        wpo == 0.0 && continue
                        for (li, gdi) in enumerate(idξ)
                            ki = pos[gdi]; ki == 0 && continue
                            v = coef*wpo*Sξ[li]
                            v == 0.0 && continue
                            for ixp in 1:nx
                                Rxy[oidx(iα,ix,ko), oidx(iαp,ixp,ki)] += v*fπb[ixp]
                            end
                        end
                    end
                end
            end
        end
    end

    # ---- assemble H = T + V + V·Rxy ,  B = kron(Nx, Syk) ----
    blk(iα) = (iα-1)*nx*ndy
    H = zeros(Float64, Ntot, Ntot); B = zeros(Float64, Ntot, Ntot)
    Iy = Matrix{Float64}(I, ndy, ndy)
    for iα in 1:nch
        λ = Int(round(α.λ[iα]))
        r = blk(iα)+1 : blk(iα)+nx*ndy
        # T block (diagonal in channel): kron(Tx_ch, Sy) + kron(Nx, Ky_λ)
        H[r,r] .+= kron(real.(Tx_ch[iα]), Syk) .+ kron(Nx, Kyk[λ])
        B[r,r] .+= kron(Nx, Syk)
    end
    # V (standalone) = kron(Vx, Sy) ; VxI = kron(Vx, I) for V·Rxy
    Vfull  = zeros(Float64, Ntot, Ntot); VxI = zeros(Float64, Ntot, Ntot)
    for iα in 1:nch, iαp in 1:nch
        Vblk = real.(Vx[iα,iαp])
        all(==(0.0), Vblk) && continue
        ro = blk(iα)+1:blk(iα)+nx*ndy; co = blk(iαp)+1:blk(iαp)+nx*ndy
        Vfull[ro,co] .+= kron(Vblk, Syk)
        VxI[ro,co]   .+= kron(Vblk, Iy)
    end
    H .+= Vfull .+ VxI*Rxy

    ev = eigen(H, B); re = sort(real.(ev.values))
    neg = re[re .< 0.0]
    return neg
end

println("="^70)
println(" 3-body MT bound state, y on HERMITE SPLINE (θ=0).  Target ≈ −8.52 MeV")
println("="^70)
for (nx,xm,ym,nyint) in ((12,16.0,16.0,14),(16,16.0,16.0,20),(16,18.0,18.0,24),(20,18.0,18.0,28))
    neg = run_bound(nx=nx, xmax=xm, ymax=ym, nyint=nyint, nq=6)
    @printf(" nx=%d box=%.0f nyint=%d  ground = %.5f MeV\n", nx, xm, nyint, neg[1])
end
println(" [all-Lagrange reference: −8.509 (16/16) → −8.516 (20/18) → −8.520 (24/20)]")
