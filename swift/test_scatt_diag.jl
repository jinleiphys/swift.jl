# Fast scattering diagnostics (NO scattering solve): localize the |S| / recoupling bug.
using LinearAlgebra, Printf
include("../general_modules/channels.jl"); using .channels
include("../general_modules/mesh.jl");     using .mesh
include("matrices_optimized.jl");          using .matrices_optimized
include("scattering.jl");                   using .Scattering

# doublet n+d channel space (same as benchmark)
fermion=true; Jtot=0.5; T=0.5; Parity=1; MT=-0.5
α = α3b(fermion, Jtot, T, Parity, 2,0, 2,0, 0.5,0.5,0.5, 0.5,0.5,0.5, MT, 1.0)

# identify deuteron channels exactly as compute_scattering_amplitude does
dch = Int[]; labels = String[]
for iα in 1:length(α.l)
    i2b = α.α2bindex[iα]
    l2b = α.α2b.l[i2b]; s12 = α.α2b.s12[i2b]; J12 = α.α2b.J12[i2b]
    if Int(round(J12))==1 && Int(round(s12))==1 && (l2b==0 || l2b==2)
        push!(dch, iα)
        push!(labels, (l2b==0 ? "³S₁" : "³D₁")*", λ=$(Int(round(α.λ[iα])))")
    end
end
n = length(dch)
println("deuteron channels (n=$n): ", labels)

function id_test(dch_, tag)
    println("\n=== identity test  T·I·T†  ($tag, n=$(length(dch_))) ===")
    U_id = Matrix{ComplexF64}(I, length(dch_), length(dch_))
    U_cs, lab = Scattering.recouple_to_channel_spin(U_id, α, dch_)
    for (key, M) in U_cs
        println("  (J=$(key[1]), π=$(key[2]))  labels=", lab[key])
        for i in 1:size(M,1)
            @printf("    "); for j in 1:size(M,2); @printf("%+.4f ", real(M[i,j])); end; println()
        end
    end
end

id_test(dch, "FULL 4 channels (³S₁+³D₁)")

# ³S₁-only deuteron channels (correct for MT: deuteron is pure ³S₁)
dch_S = [iα for iα in dch if α.α2b.l[α.α2bindex[iα]] == 0]
id_test(dch_S, "³S₁ only")
