module threebodybound 
 include("matrices.jl")
 using .matrices
 using LinearAlgebra
 using Printf
 export ThreeBody_Bound


 function ThreeBody_Bound(α, grid, potname,e2b)
    
    println("\n" * "="^60)
    println("         TIMING ANALYSIS: ThreeBody_Bound")
    println("="^60)
    
    # Time matrix construction
    println("Building matrices...")
    t_start = time()
    
    t_rxy_start = time()
    Rxy=Rxy_matrix(α, grid)
    t_rxy = time() - t_rxy_start
    
    t_t_start = time()
    T=T_matrix(α,grid)
    t_t = time() - t_t_start
    
    t_v_start = time()
    V=V_matrix(α, grid, potname)
    t_v = time() - t_v_start
    
    t_h_start = time()
    H=V*Rxy+T+V
    t_h = time() - t_h_start
    
    t_b_start = time()
    B=Bmatrix(α,grid)
    t_b = time() - t_b_start
    
    t_matrices = time() - t_start
    @printf("Total matrix construction time: %.4f seconds\n", t_matrices)
    
    # Time eigenvalue decomposition
    println("\nSolving eigenvalue problem...")
    t_eigen_start = time()
    eigenvalues, eigenvectors = eigen(H,B)
    t_eigen = time() - t_eigen_start
    @printf("Eigenvalue decomposition: %.4f seconds\n", t_eigen)
    

    # Extract the bound state energies and wave functions
    bound_states = []
    
    println("\n" * "="^60)
    println("         THREE-BODY BOUND STATE ANALYSIS")
    println("="^60)
    println("Two-body threshold energy: $(round(e2b[1], digits=6)) MeV")
    println("Searching for three-body bound states...")
    
    t_analysis_start = time()
    bound_count = 0
    t_wavefunction_total = 0.0
    t_norm_total = 0.0
    t_verification_total = 0.0
    
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
            t_wf_start = time()
            eigenvec = eigenvectors[:, i]
            
            # For generalized eigenvalue problem, normalize eigenvector properly
            # The eigenvector should satisfy: eigenvec' * B * eigenvec = 1
            B_norm = real(eigenvec' * B * eigenvec)
            if B_norm > 1e-12
                eigenvec = eigenvec / sqrt(B_norm)
            end
            
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
            t_wf = time() - t_wf_start
            t_wavefunction_total += t_wf
            
            # Compute normalization
            t_norm_start = time()
            norm_total = 0.0
            for ich in 1:α.nchmax
                for ix in 1:grid.nx
                    for iy in 1:grid.ny
                        norm_total += abs2(ψ[ix, iy, ich]) * grid.dxi[ix] * grid.dyi[iy]
                    end
                end
            end
            t_norm = time() - t_norm_start
            t_norm_total += t_norm
            
            println("  Wave function norm: $(round(sqrt(norm_total), digits=6))")
            println("  B-orthogonal norm: $(round(sqrt(real(eigenvec' * B * eigenvec)), digits=6))")
            
            # Verify binding energy by computing expectation value ⟨ψ|H|ψ⟩ and ⟨ψ|T|ψ⟩
            # Since eigenvector is B-normalized: eigenvec' * B * eigenvec = 1
            t_verify_start = time()
            H_expectation = real(eigenvec' * H * eigenvec)
            T_expectation = real(eigenvec' * T * eigenvec)
            energy_difference = abs(H_expectation - energy_real)
            t_verify = time() - t_verify_start
            t_verification_total += t_verify
            
            println("  Eigenvalue energy: $(round(energy_real, digits=6)) MeV")
            println("  ⟨ψ|H|ψ⟩ energy: $(round(H_expectation, digits=6)) MeV")
            println("  ⟨ψ|T|ψ⟩ kinetic energy: $(round(T_expectation, digits=6)) MeV")
            println("  Energy difference: $(round(energy_difference, digits=8)) MeV")
            
            if energy_difference < 1e-6
                println("  ✓ Energy verification: PASSED")
            else
                println("  ⚠ Energy verification: WARNING - Large difference!")
            end
            
            push!(bound_states, (eigenvalues[i], eigenvectors[:, i], ψ))
        end
    end
    
    t_analysis = time() - t_analysis_start
    
    println("\n" * "="^60)
    println("SUMMARY: Found $bound_count three-body bound state(s)")
    if bound_count > 0
        println("Three-body binding energies (MeV): ", [round(real(state[1]), digits=6) for state in bound_states])
    else
        println("No three-body bound states found below two-body threshold.")
    end
    println("="^60)
    
    # Print detailed timing summary
    println("\n" * "="^60)
    println("         DETAILED TIMING BREAKDOWN")
    println("="^60)
    @printf("Total matrices:          %.4f seconds (%.1f%%)\n", t_matrices, 100*t_matrices/(t_matrices+t_eigen+t_analysis))
    @printf("\n")
    @printf("Eigenvalue solution:     %.4f seconds (%.1f%%)\n", t_eigen, 100*t_eigen/(t_matrices+t_eigen+t_analysis))
    @printf("\n")
    @printf("Bound state analysis breakdown:\n")
    @printf("  Wave function comp:    %.4f seconds (%.1f%%)\n", t_wavefunction_total, 100*t_wavefunction_total/(t_matrices+t_eigen+t_analysis))
    @printf("  Normalization:         %.4f seconds (%.1f%%)\n", t_norm_total, 100*t_norm_total/(t_matrices+t_eigen+t_analysis))
    @printf("  Energy verification:   %.4f seconds (%.1f%%)\n", t_verification_total, 100*t_verification_total/(t_matrices+t_eigen+t_analysis))
    @printf("Total analysis:          %.4f seconds (%.1f%%)\n", t_analysis, 100*t_analysis/(t_matrices+t_eigen+t_analysis))
    @printf("\n")
    t_total = t_matrices + t_eigen + t_analysis
    @printf("TOTAL FUNCTION TIME:     %.4f seconds\n", t_total)
    @printf("\nMost time-consuming operations:\n")
    timings = [("Matrix construction", t_matrices), ("Eigenvalue decomposition", t_eigen), ("Bound state analysis", t_analysis)]
    sort!(timings, by=x->x[2], rev=true)
    for (i, (name, time_val)) in enumerate(timings)
        @printf("  %d. %s: %.4f seconds (%.1f%%)\n", i, name, time_val, 100*time_val/t_total)
    end
    println("="^60)

    return bound_states

 end 

end 