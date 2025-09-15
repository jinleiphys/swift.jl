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
const m=1.0079713395678829 # amu
const ħ=197.3269718 # MeV. fm

export Rxy_matrix, T_matrix, V_matrix, Bmatrix

# 1.008665 amu for neutron  amu=931.49432 MeV

function Rxy_matrix(α, grid)
    # the channel index can be computed by i(iα,ix, iy) = (iα-1) * grid.nx * grid.ny + (ix-1)*grid.ny + iy
    Rxy_32 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny) # Initialize Rxy_32 matrix
    Rxy_31 = zeros(Complex{Float64}, α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny) # Initialize Rxy_31 matrix
    
    # Fixed typo in function name
    Gαα = computeGcoefficient(α, grid)
       
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
    
    Rxy = Rxy_31 + Rxy_32
    
    return Rxy,Rxy_31, Rxy_32
end




 function T_matrix(α,grid) 
"""
α.nchmax is the maximum number of α channel index, α0 is the α parameter in Laguerre function 
"""
 
 # Compute correct overlap matrix for y-direction (non-orthogonal basis functions)
 Ny = zeros(grid.ny, grid.ny)
 for i in 1:grid.ny
     for j in 1:grid.ny
         if i == j
             Ny[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
         else
             Ny[i,j] = (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
         end
     end
 end

 # Elegant Kronecker product sum structure: ∑_α δ_{α,α} I_α ⊗ Tx^α ⊗ Ny
 Tx_matrix = zeros(α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
 
 for i in 1:α.nchmax
     # Compute Tx^α for channel α with its specific l[α]
     Tx_alpha = Tx(grid.nx, grid.xx, grid.α, α.l[i])
     Tx_alpha .= Tx_alpha .* ħ^2 / m / amu / grid.hsx^2
     
     # Create channel selector matrix: δ_{α,α} I_α (only α-th diagonal element is 1)
     I_alpha = zeros(α.nchmax, α.nchmax)
     I_alpha[i, i] = 1.0
     
     # Add this channel's contribution: δ_{α,α} I_α ⊗ Tx^α ⊗ Ny
     Tx_matrix += I_alpha ⊗ Tx_alpha ⊗ Ny
 end
 
 # Compute correct overlap matrix for x-direction (non-orthogonal basis functions)
 Nx = zeros(grid.nx, grid.nx)
 for i in 1:grid.nx
     for j in 1:grid.nx
         if i == j
             Nx[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
         else
             Nx[i,j] = (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
         end
     end
 end
 
 # Elegant Kronecker product sum structure: ∑_α δ_{α,α} I_α ⊗ Nx ⊗ Ty^α
 Ty_matrix = zeros(α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
 
 for i in 1:α.nchmax
     # Compute Ty^α for channel α with its specific λ[α]
     Ty_alpha = Tx(grid.ny, grid.yy, grid.α, α.λ[i])
     Ty_alpha .= Ty_alpha .* ħ^2 * 0.75 / m / amu / grid.hsy^2
     
     # Create channel selector matrix: δ_{α,α} I_α (only α-th diagonal element is 1)
     I_alpha = zeros(α.nchmax, α.nchmax)
     I_alpha[i, i] = 1.0
     
     # Add this channel's contribution: δ_{α,α} I_α ⊗ Nx ⊗ Ty^α
     Ty_matrix += I_alpha ⊗ Nx ⊗ Ty_alpha
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
    # Get nuclear potential matrix
    v12 = pot_nucl(α, grid, potname)

    # Compute correct overlap matrix for y-direction (non-orthogonal basis functions)
    Ny = zeros(grid.ny, grid.ny)
    for i in 1:grid.ny
        for j in 1:grid.ny
            if i == j
                Ny[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            else
                Ny[i,j] = (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            end
        end
    end

    # Express V_matrix using diagonal decomposition with distinct matrices
    # V_matrix = ∑_{α₃,α₃'} δ_{allowed}(α₃',α₃) [P^{α₃',α₃}] ⊗ [V_{kₓ}^{α₃',α₃}] ⊗ [N_{k_y}]
    # where P^{α₃',α₃} is the channel projection matrix with 1 at (α₃',α₃) and 0 elsewhere
    
    Vmatrix = zeros(α.nchmax*grid.nx*grid.ny, α.nchmax*grid.nx*grid.ny)
    
    for j in 1:α.nchmax  # α₃'
        for i in 1:α.nchmax  # α₃
            # Check if this channel pair couples
            if α.T12[i] != α.T12[j]
                continue  # Skip if T₁₂ ≠ T₁₂'
            end
            if α.λ[i] != α.λ[j]
                continue  # Skip if λ₃ ≠ λ₃'
            end
            if α.J3[i] != α.J3[j]
                continue  # Skip if J₃ ≠ J₃'
            end
            if α.s12[i] != α.s12[j]
                continue  # Skip if s₁₂ ≠ s₁₂'
            end
            if α.J12[i] != α.J12[j]
                continue  # Skip if J₁₂ ≠ J₁₂'
            end
            
            # Create channel projection matrix P^{α₃',α₃}
            P_channel = zeros(α.nchmax, α.nchmax)
            P_channel[i, j] = 1.0
            
            # Extract the potential matrix block V^{ij}_x for this channel pair
            V_x_ij = zeros(grid.nx, grid.nx)
            
            T12 = α.T12[i]
            # Sum over m_{t₁₂} to build the potential block
            nmt12_max = Int(2 * T12)
            for nmt12 in -nmt12_max:2:nmt12_max
                mt12 = nmt12 / 2.0
                mt3 = α.MT - mt12
                
                if abs(mt3) > α.t3
                    continue
                end
                
                cg1 = clebschgordan(T12, mt12, α.t3, mt3, α.T[i], α.MT)      
                cg2 = clebschgordan(T12, mt12, α.t3, mt3, α.T[j], α.MT)
                cg_coefficient = cg1 * cg2
                
                if abs(cg_coefficient) < 1e-10
                    continue
                end
                # Select the appropriate potential matrix element based on mt12
                # mt12 = 0 corresponds to np pair (isospin singlet/triplet mixed)  
                # mt12 ≠ 0 corresponds to nn or pp pair
                if mt12 == 0
                    V_x_ij += v12[:, :, α.α2bindex[i], α.α2bindex[j], 1] * cg_coefficient
                else
                    V_x_ij += v12[:, :, α.α2bindex[i], α.α2bindex[j], 2] * cg_coefficient
                end
            end
            
            # Add this channel pair's contribution: P^{α₃',α₃} ⊗ V_{kₓ}^{α₃',α₃} ⊗ N_{k_y}
            Vmatrix += P_channel ⊗ V_x_ij ⊗ Ny
        end
    end    

    
    return Vmatrix
end

function pot_nucl(α, grid, potname)
    # Compute the nuclear potential matrix
    # Parameters:
    # α: channel index
    # grid: grid object containing nx, xi, and other parameters
    # proton m1=+1/2  neutron m2=-1/2
    # for the current function, I only consider the local potential(AV8,NIJM,REID,AV14,AV18), for the non-local potential, one needs to modify this function 
    v12 = zeros(grid.nx, grid.nx, α.α2b.nchmax, α.α2b.nchmax, 2)  # Initialize potential matrix the last dimension is for the isospin 1 for np pair and 2 for nn(MT<0) or pp pair(MT>0)

    for j in 1:α.α2b.nchmax
        for i in 1:α.α2b.nchmax
            if checkα2b(i, j, α)
                li=[α.α2b.l[i]]
                # Compute the potential matrix elements
                if Int(α.α2b.J12[i]) == 0  # Special case: J12=0
                    if α.α2b.l[i] != α.α2b.l[j]
                        continue  # Skip if l[i] != l[j] for J12=0 case
                    end
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir],li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end
                elseif Int(α.α2b.J12[i]) == α.α2b.l[i]  # Uncoupled states: J12=l (but not J12=0)
                    if α.α2b.l[i] != α.α2b.l[j]
                        error("error: the channel is not allowed")
                    end 
                    for ir in 1:grid.nx  # note that for nonlocal potential, additional loops is needed
                        v = potential_matrix(potname, grid.xi[ir],li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                        v12[ir, ir, i, j, 1] = v[1, 1]
                        if α.MT > 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                            v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                        elseif α.MT < 0
                            v = potential_matrix(potname, grid.xi[ir], li, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                            v12[ir, ir, i, j, 2] = v[1, 1]
                        # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                        end
                    end
                else
                    # For coupled channels, both i and j should have the same J12 due to delta function constraint
                    J12_val = Int(α.α2b.J12[i])  # Could also use α.α2b.J12[j] since they should be equal
                    l = [J12_val-1, J12_val+1]
                    for ir in 1:grid.nx  
                        if α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 1] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 1]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 2] + VCOUL_point(grid.xi[ir], 1.0) # for pp pair
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[2, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val-1) && α.α2b.l[j] == (J12_val+1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[1, 2]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[1, 2] 
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
                                v12[ir, ir, i, j, 2] = v[1, 2]
                            # else: α.MT == 0, only compute v12[ir, ir, i, j, 1], leave v12[ir, ir, i, j, 2] as zero
                            end
                        elseif α.α2b.l[i] == (J12_val+1) && α.α2b.l[j] == (J12_val-1) 
                            v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 0)
                            v12[ir, ir, i, j, 1] = v[2, 1]
                            if α.MT > 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), 1) # for pp pair
                                v12[ir, ir, i, j, 2] = v[2, 1]  
                            elseif α.MT < 0
                                v = potential_matrix(potname, grid.xi[ir], l, Int(α.α2b.s12[i]), Int(α.α2b.J12[i]), Int(α.α2b.T12[i]), -1) # for nn pair
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

 function Bmatrix(α,grid)
    # compute the B matrix for the Generalized eigenvalue problem
    Iα = Matrix{Float64}(I, α.nchmax, α.nchmax)
    Nx=zeros(grid.nx, grid.nx)
    Ny=zeros(grid.ny, grid.ny)
    for i in 1:grid.nx
        for j in 1:grid.nx
            if i == j
                Nx[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            else
                Nx[i,j] = (-1.)^(j-i)/sqrt(grid.xx[i]*grid.xx[j])
            end
        end
    
    end 

    for i in 1:grid.ny
        for j in 1:grid.ny
            if i == j
                Ny[i,j] = 1 + (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            else
                Ny[i,j] = (-1.)^(j-i)/sqrt(grid.yy[i]*grid.yy[j])
            end
        end
    
    end

    Bmatrix = Iα ⊗ Nx ⊗ Ny

    return Bmatrix


 end 


 function checkα2b(i,j,α)
    # Check if the two-body channels are allowed for potential coupling
    # The two-body potential should only couple channels with identical quantum numbers
    if α.α2b.T12[i] == α.α2b.T12[j] && α.α2b.s12[i] == α.α2b.s12[j] && α.α2b.J12[i] == α.α2b.J12[j] && (-1)^α.α2b.l[i] == (-1)^α.α2b.l[j]
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