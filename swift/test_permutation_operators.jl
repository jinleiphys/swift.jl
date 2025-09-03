module TestPermutationOperators

include("matrices.jl")
using .matrices
using LinearAlgebra

export test_permutation_identity

"""
Test module for permutation operators P+ and P- to verify the identity:
P+ * P- * ψ = P- * P+ * ψ = ψ

This tests that the permutation operators form a valid representation
and that their composition returns to the identity.
"""
function test_permutation_identity(α, grid, ψ)
    println("Testing permutation operator identity...")
    println("Input wavefunction size: $(size(ψ))")
    
    # Create test wavefunction with only first channel (ich=1) non-zero
    ψ_test = zeros(ComplexF64, α.nchmax * grid.nx * grid.ny)
    
    # Copy only the first channel from input ψ
    channel_size = grid.nx * grid.ny
    ψ_test[1:channel_size] = ψ[1:channel_size]
    
    println("Test wavefunction: only channel 1 active, others zero")
    println("Non-zero elements in channel 1: $(count(x -> abs(x) > 1e-12, ψ_test[1:channel_size]))")
    
    # Get the Rxy matrices
    Rxy, Rxy_minus, Rxy_plus = Rxy_matrix(α, grid)
    
    println("Rxy_plus (P+) matrix size: $(size(Rxy_plus))")
    println("Rxy_minus (P-) matrix size: $(size(Rxy_minus))")
    
    # Test composition: P- * P+ * ψ
    ψ_temp1 = Rxy_plus * ψ_test
    ψ_out1 = Rxy_minus * ψ_temp1
    
    # Test composition: P+ * P- * ψ  
    ψ_temp2 = Rxy_minus * ψ_test
    ψ_out2 = Rxy_plus * ψ_temp2
    
    # Check if both compositions equal original test wavefunction
    diff1 = norm(ψ_out1 - ψ_test)
    diff2 = norm(ψ_out2 - ψ_test)
    diff12 = norm(ψ_out1 - ψ_out2)
    
    println("\nPermutation identity test results:")
    println("||P- * P+ * ψ - ψ|| = $(diff1)")
    println("||P+ * P- * ψ - ψ|| = $(diff2)")
    println("||P- * P+ * ψ - P+ * P- * ψ|| = $(diff12)")
    
    # Define tolerance for numerical precision
    tol = 1e-12
    
    identity_test1 = diff1 < tol
    identity_test2 = diff2 < tol
    commutativity_test = diff12 < tol
    
    println("\nTest results (tolerance = $(tol)):")
    println("P- * P+ = I: $(identity_test1 ? "PASS" : "FAIL")")
    println("P+ * P- = I: $(identity_test2 ? "PASS" : "FAIL")")
    println("P+ and P- commute: $(commutativity_test ? "PASS" : "FAIL")")
    
    if identity_test1 && identity_test2 && commutativity_test
        println("\n✓ All permutation operator tests PASSED")
        return true
    else
        println("\n✗ Some permutation operator tests FAILED")
        return false
    end
end

end # module