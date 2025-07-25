module matrices 
using Kronecker
include("../NNpot/nuclear_potentials.jl")
using .NuclearPotentials
using WignerSymbols
include("laguerre.jl")
using .Laguerre
include("Gcoefficient.jl")
using .Gcoefficient
using LinearAlgebra

const amu= 931.49432 # MeV
const m=1.008665 # amu
const ħ=197.3269718 # MeV. fm

export Rxy_matrix, T_matrix, V_matrix

# 1.008665 amu for neutron  amu=931.49432 MeV

function Rxy_matrix(α, grid)
    # the channel index can be computed by i(iα,ix, iy) = (iα-1) * grid.nx * grid.ny + (ix-1)*grid.ny + iy
    Rxy_32 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny) # Initialize Rxy_32 matrix
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny) # Initialize Rxy_31 matrix
    
    # Fixed typo in function name
    Gαα = computeGcoefficient(α, grid)
    
    # compute the Rxy matrix from α2 to α3
    a = -0.5; b = -1.0; c = 0.75; d = -0.5
    for ix in 1:grid.nx
        xa = grid.xi[ix]
        for iy in 1:grid.ny
            ya = grid.yi[iy]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]
                dcosθ = grid.dcosθi[iθ]
                πb = sqrt(a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ)
                ξb = sqrt(c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ)
                
                fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)
                
                for iα in 1:α.nchmax
                    i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
                    for iαp in 1:α.nchmax
                        adj_factor = dcosθ * Gαα[iθ, iy, ix, iα, iαp, 2] * xa * ya / (πb * ξb * grid.ϕx[ix] * grid.ϕy[iy]) 
                        for ixp in 1:grid.nx
                            for iyp in 1:grid.ny
                                ip = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny + iyp
                                Rxy_32[i, ip] += adj_factor * fπb[ixp] * fξb[iyp]
                            end
                        end
                    end
                end
            end
        end
    end
    
    # compute the Rxy matrix from α1 to α3
    a = -0.5; b = 1.0; c = -0.75; d = -0.5
    for ix in 1:grid.nx
        xa = grid.xi[ix]
        for iy in 1:grid.ny
            ya = grid.yi[iy]
            for iθ in 1:grid.nθ
                cosθ = grid.cosθi[iθ]
                dcosθ = grid.dcosθi[iθ]
                πb = sqrt(a^2 * xa^2 + b^2 * ya^2 + 2*a*b*xa*ya*cosθ)
                ξb = sqrt(c^2 * xa^2 + d^2 * ya^2 + 2*c*d*xa*ya*cosθ)
                
                
                fπb = lagrange_laguerre_regularized_basis(πb, grid.xi, grid.ϕx, grid.α, grid.hsx)
                fξb = lagrange_laguerre_regularized_basis(ξb, grid.yi, grid.ϕy, grid.α, grid.hsy)
                
                for iα in 1:α.nchmax
                    i = (iα-1)*grid.nx*grid.ny + (ix-1)*grid.ny + iy
                    for iαp in 1:α.nchmax
                        adj_factor = dcosθ * Gαα[iθ, iy, ix, iα, iαp, 1] * xa * ya / (πb * ξb * grid.ϕx[ix] * grid.ϕy[iy] )
                        for ixp in 1:grid.nx
                            for iyp in 1:grid.ny
                                ip = (iαp-1)*grid.nx*grid.ny + (ixp-1)*grid.ny + iyp
                                Rxy_31[i, ip] += adj_factor * fπb[ixp] * fξb[iyp]
                            end
                        end
                    end
                end
            end
        end
    end
    
    Rxy = Rxy_31 + Rxy_32
    
    return Rxy
end




 function T_matrix(α,grid) 
"""
α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
"""
 Tαx = zeros(α.nchmax*grid.nx,α.nchmax*grid.nx)  # Initialize Tαx matrix
 Iy = Matrix{Float64}(I, grid.ny, grid.ny)

 for i in 1:α.nchmax

    T = Tx(grid.nx,grid.xx,grid.α,α.l[i])  
    T .= T .* ħ^2 / m / amu / grid.hsx^2
    row_start = (i-1)*grid.nx + 1
    row_end = i*grid.nx
    col_start = (i-1)*grid.nx + 1
    col_end = i*grid.nx
    Tαx[row_start:row_end, col_start:col_end] = T

 end 

 Tx_matrix = Tαx ⊗ Iy
 
 Ty_matrix = zeros(α.nchmax*grid.nx*grid.ny,α.nchmax*grid.nx*grid.ny)  # Initialize Ty_matrix
 i=0
 for iα in 1:α.nchmax
    for ix in 1:grid.nx 
       i += 1
       T= Tx(grid.ny,grid.yy,grid.α,α.λ[iα])
       T .= T .* ħ^2 * 0.75 / m / amu /grid.hsy^2

       row_start = (i-1)*grid.ny + 1
       row_end = i*grid.ny
       col_start = (i-1)*grid.ny + 1
       col_end = i*grid.ny
       Ty_matrix[row_start:row_end, col_start:col_end] = T

    end 
 end 

 Tmatrix = Tx_matrix + Ty_matrix


 return Tmatrix
 end 


 function Tx(nx,xi,α0,l)
    # Compute the T matrix 
    # Parameters:
    # nx: number of points in the radial mesh
    # xi: radial mesh points
    # α0: parameter for the Laguerre function
    # l: angular momentum quantum number
    T = zeros(nx,nx)  # Initialize T matrix

    for i in 1:nx
        for j in 1:nx
            if i == j
                # Diagonal elements 
                T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j])) + 
                         l * (l + 1) / xi[i]^2

                # T[i,j] = (-1.0 / (12.0 * xi[i]^2)) * (xi[i]^2 - 2.0 * (2.0 * nx + α0 + 1.0) * xi[i] + α0^2 - 4.0) + 
                #          l * (l + 1) / xi[i]^2
            else
                # Off-diagonal elements 
                T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) - 
                         (-1)^(i-j) / (4 * sqrt(xi[i] * xi[j]))

                # T[i,j] = (-1.0)^(i-j) * (xi[i] + xi[j]) / (sqrt(xi[i] * xi[j]) * (xi[i] - xi[j])^2) 
            end
        end
    end


    return T

 end # end function Tx

 function V_matrix(α, grid, potname)
    # Initialize matrix for storing potential energy
    Vαx = zeros(α.nchmax * grid.nx, α.nchmax * grid.nx)
    
    # Get nuclear potential matrix
    v12 = pot_nucl(α, grid, potname)

    Iy = Matrix{Float64}(I, grid.ny, grid.ny)
    
    # Implement the three-body matrix element based on the given equation:
    # V_{α₃,α₃'}(x₃,x₃') = Σ_{m_{t₁₂}} ⟨T₁₂ m_{t₁₂} t₃(M_T - m_{t₁₂}) | T M_T⟩ 
    #                      × ⟨T₁₂ m_{t₁₂} t₃(M_T - m_{t₁₂}) | T' M_T⟩ δ_{T₁₂,T₁₂'} V₁₂^{T₁₂,m_{t₁₂}}(x₃,x₃')
    
    for j in 1:α.nchmax  # α₃'
        for i in 1:α.nchmax  # α₃
            # Calculate start and end indices for blocks in the matrix
            row_start = (i - 1) * grid.nx + 1
            row_end = i * grid.nx
            col_start = (j - 1) * grid.nx + 1
            col_end = j * grid.nx
            
            # Check if T₁₂ = T₁₂' (Kronecker delta constraint)
            if α.T12[i] != α.T12[j]
                continue  # Skip if T₁₂ ≠ T₁₂'
            end
            
            T12 = α.T12[i]  # = α.T12[j] due to delta function
            
            # Sum over m_{t₁₂} (magnetic quantum number of two-body subsystem)
            nmt12_max = Int(2 * T12)
            for nmt12 in -nmt12_max:2:nmt12_max
                mt12 = nmt12 / 2.0
                mt3 = α.MT - mt12  # From conservation: m_{t₁₂} + m_{t₃} = M_T
                
                # Check if |m_{t₃}| ≤ t₃ (physical constraint)
                if abs(mt3) > α.t3
                    continue
                end
                
                # Calculate Clebsch-Gordan coefficients
                # ⟨T₁₂ m_{t₁₂} t₃ m_{t₃} | T M_T⟩
                cg1 = clebschgordan(T12, mt12, α.t3, mt3, α.T, α.MT)
                # ⟨T₁₂ m_{t₁₂} t₃ m_{t₃} | T' M_T⟩  (T' = T for same total state)
                cg2 = clebschgordan(T12, mt12, α.t3, mt3, α.T, α.MT)
                
                # Combined coefficient: cg1 × cg2
                cg_coefficient = cg1 * cg2
                
                if abs(cg_coefficient) < 1e-10  # Skip negligible contributions
                    continue
                end
                
                # Select appropriate potential matrix element V₁₂^{T₁₂,m_{t₁₂}}
                # Based on m_{t₁₂} to determine isospin channel
                if mt12 == 0
                    # Isospin-0 channel (np pair)
                    Vαx[row_start:row_end, col_start:col_end] += v12[:, :, i, j, 1] * cg_coefficient
                else
                    # Isospin-1 channel (pp or nn pair)
                    Vαx[row_start:row_end, col_start:col_end] += v12[:, :, i, j, 2] * cg_coefficient
                end
            end
        end
    end

    Vmatrix = Vαx ⊗ Iy  # Kronecker product with identity matrix    

    
    return Vmatrix
end

function pot_nucl(α, grid, potname)
    # Compute the nuclear potential matrix
    # Parameters:
    # α: channel index
    # grid: grid object containing nx, xi, and other parameters
    # proton m1=+1/2  neutron m2=-1/2
    # for the current function, I only consider the local potential(AV8,NIJM,REID,AV14,AV18), for the non-local potential, one needs to modify this function 
    v12 = zeros(grid.nx, grid.nx, α.nchmax, α.nchmax, 2)  # Initialize potential matrix the last dimension is for the isospin 1 for np pair and 2 for nn(MT<0) or pp pair(MT>0)

    for j in 1:α.nchmax
        for i in 1:α.nchmax
            if checkα(i, j, α)
                li=[α.l[i]]
                # Compute the potential matrix elements
                if Int(α.J12[i]) == 0
                    if α.l[i] != α.l[j]
                        error("error: the channel is not allowed")
                    end 
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir],li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end 
                    
                elseif Int(α.J12[i]) == α.l[i]
                    if α.l[i] != α.l[j]
                        error("error: the channel is not allowed")
                    end
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir], li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end
                else
                    l = [Int(α.J12[i])-1, Int(α.J12[i])+1]
                    for ir in 1:grid.nx  
                        if α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.l[i] == Int(α.J12[i]-1) && α.l[j] == Int(α.J12[i]+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 2] 
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.l[i] == Int(α.J12[i]+1) && α.l[j] == Int(α.J12[i]-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 1]  
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.s12[i]), Int(α.J12[i]), Int(α.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 1]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        end
                    end 
                end 
            end
        end
    end
    
    return v12  
end


 
 function checkα(i,j,α)

    # Check if the channel is allowed
    if (-1)^α.l[i] == (-1)^α.l[j] && Int(α.s12[i]*2) == Int(α.s12[j]*2) && Int(α.J12[i]*2) == Int(α.J12[j]*2)
        return true
    else
        return false
    end
 end 


 function VCOUL_point(R, z12)   # use to compute the Coulomb potential
    # Constants
    e2 = 1.43997  # Coulomb constant in appropriate units
    
    # Calculations
    aux = e2 * z12
    vcoul_point = 0.0
    
    # Early return if z12 is very small
    if (z12 < 1e-4)
        return vcoul_point
    end
    
    # Compute Coulomb potential
    vcoul_point = aux / R
    
    return vcoul_point
end





end # end module matrices