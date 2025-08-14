module threebodybound 
 include("matrices.jl")
 using .matrices
 using LinearAlgebra
 export ThreeBody_Bound


 function ThreeBody_Bound(α, grid, potname,e2b)

    Rxy=Rxy_matrix(α, grid)
    T=T_matrix(α,grid)
    V=V_matrix(α, grid, potname)
    H=V*Rxy+T+V

    eigenvalues, eigenvectors = eigen(H)
    

    # Extract the bound state energies and wave functions
    bound_states = []
    
    println("\n" * "="^60)
    println("         THREE-BODY BOUND STATE ANALYSIS")
    println("="^60)
    println("Two-body threshold energy: $(round(e2b[1], digits=6)) MeV")
    println("Searching for three-body bound states...")
    
    bound_count = 0
    for i in 1:grid.nx
        energy_real = real(eigenvalues[i])
        if energy_real < 0.0 && energy_real < e2b[1]
            bound_count += 1
            energy_imag = imag(eigenvalues[i])
            
            println("\nThree-body Bound State #$bound_count:")
            println("  Binding Energy: $(round(energy_real, digits=6)) MeV")
            if abs(energy_imag) > 1e-10
                println("  Imaginary part: $(round(energy_imag, digits=8)) MeV")
            end
            println("  Energy gain from 2-body: $(round(energy_real - e2b[1], digits=6)) MeV")
            
            # Compute wave function ψ(x,y,α) based on Laguerre basis functions
            eigenvec = eigenvectors[:, i]
            ψ = zeros(ComplexF64, grid.nx, grid.ny, α.nchmax)
            
            # Extract coefficients and compute wave function
            for ich in 1:α.nchmax
                for ix in 1:grid.nx
                    for iy in 1:grid.ny
                        idx = (ich-1)*grid.nx*grid.ny + (iy-1)*grid.nx + ix
                        if idx <= length(eigenvec)
                            ψ[ix, iy, ich] = grid.ϕx[ix] * grid.ϕy[iy] * eigenvec[idx]
                        end
                    end
                end
            end
            
            # Compute normalization
            norm_total = 0.0
            for ich in 1:α.nchmax
                for ix in 1:grid.nx
                    for iy in 1:grid.ny
                        norm_total += abs2(ψ[ix, iy, ich]) * grid.dxi[ix] * grid.dyi[iy]
                    end
                end
            end
            
            println("  Wave function norm: $(round(sqrt(norm_total), digits=6))")
            
            # Verify binding energy by computing expectation value ⟨ψ|H|ψ⟩
            H_expectation = real(eigenvec' * H * eigenvec)
            energy_difference = abs(H_expectation - energy_real)
            
            println("  Eigenvalue energy: $(round(energy_real, digits=6)) MeV")
            println("  ⟨ψ|H|ψ⟩ energy: $(round(H_expectation, digits=6)) MeV")
            println("  Energy difference: $(round(energy_difference, digits=8)) MeV")
            
            if energy_difference < 1e-6
                println("  ✓ Energy verification: PASSED")
            else
                println("  ⚠ Energy verification: WARNING - Large difference!")
            end
            
            push!(bound_states, (eigenvalues[i], eigenvectors[:, i], ψ))
        end
    end
    
    println("\n" * "="^60)
    println("SUMMARY: Found $bound_count three-body bound state(s)")
    if bound_count > 0
        println("Three-body binding energies (MeV): ", [round(real(state[1]), digits=6) for state in bound_states])
    else
        println("No three-body bound states found below two-body threshold.")
    end
    println("="^60)

    return bound_states

 end 

end 