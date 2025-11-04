using WGLMakie  # Web-based Makie (opens in browser)
using LinearAlgebra

# Set BLAS threads for performance
BLAS.set_num_threads(Sys.CPU_THREADS)

# Include necessary modules
include("general_modules/channels.jl")
include("general_modules/mesh.jl")
include("swift/twobody.jl")
include("swift/MalflietTjon.jl")

using .channels
using .mesh
using .twobodybound
using .MalflietTjon

# [Rest of the code is exactly the same as swift.jl but uses WGLMakie]
# Just change the first line from GLMakie to WGLMakie

println("="^70)
println("    SWIFT: Three-Body Nuclear Structure Calculator")
println("    Interactive GUI using WGLMakie (Browser-based)")
println("="^70)
println()
println("This version opens in your web browser.")
println("Install WGLMakie first: julia -e 'using Pkg; Pkg.add(\"WGLMakie\")'")
println()
