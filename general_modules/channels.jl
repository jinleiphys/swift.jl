module channels
using Printf
# export lmax, lmin, λmax, λmin, s1, s2, s3, Jmin, Jmax, t1, t2, t3, Tmin, Tmax, MT
export  α, α3b,λmax,lmax

# channel parameters
lmax = 0 # maximum l
lmin = 0 # minimum l
λmax = 0 # maximum λ
λmin = 0 # minimum λ
s1 = 0.0 # spin of particle 1
s2 = 0.0 # spin of particle 2
s3 = 0.0 # spin of particle 3
# for bound state J=Jmin=Jmax
Jmin = 0.0 # minimum J
Jmax = 0.0 # maximum J
t1 = 0.0 # isospin of particle 1
t2 = 0.0 # isospin of particle 2
t3 = 0.0 # isospin of particle 3
Tmin = 0.0 # minimum T
Tmax = 0.0 # maximum T
MT = 0.0 # third component of T, fix for the given system by charge conservation

mutable struct nch3b # channel index for the three body coupling
    nchmax::Int # maximum number of channels
    l::Vector{Int}
    s12::Vector{Float64}
    J12::Vector{Float64}
    λ::Vector{Int}
    s3::Vector{Float64}
    J3::Vector{Float64}
    J::Vector{Float64}
    T12::Vector{Float64}
    T::Vector{Float64}
    
    # Constructor with default initialization
    function nch3b()
        new(0, Int[], Float64[], Float64[], Int[], Float64[],Float64[], Float64[], Float64[], Float64[])
    end
end

mutable struct α_bar # channel index for the three body coupling
    αin:: Vector{Int} # channel index 
    αout:: Vector{Int} # channel index

    # Constructor with default initialization
    function α_bar()
        new(Int[], Int[])
    end
end

# Renamed to avoid conflict with the type name
global α = nch3b()
global αbar = α_bar()

# channel index of the three body channels
#|(l_{12} (s_1 s_2) s_{12}) J_{12}, (\lambda_3 s_3) J_3, J; (t_1 t_2) T_{12}, t_3, T M_T\rangle.
# Function to count and create three-body channels
function α3b(J,T,parity)
    global α

    
    # First pass to count channels
    nch_count = 0
    
    for l in lmin:lmax
        for ns in Int(2*(s1-s2)):2:Int(2*(s1+s2))
            s12 = ns/2.0
            for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))  # Fixed min/max calculation
                J12 = nJ12/2.0
                for λ in λmin:λmax
                    if (-1)^(l+λ) != parity
                        continue
                    end
                    for nJ3 in Int(2*abs(λ-s3)):2:Int(2*(λ+s3))  # Fixed min/max calculation
                        J3 = nJ3/2.0
                        for nJ in Int(2*abs(J12-J3)):2:Int(2*(J12+J3))  # Fixed min/max calculation
                            if nJ != Int(2*J)
                                continue
                            end
                            for nT12 in Int(2*abs(t1-t2)):2:Int(2*(t1+t2))
                                T12 = nT12/2.0
                                if (-1)^(l+s12+T12) != -1
                                    continue
                                end
                                for nT in Int(2*abs(T12-t3)):2:Int(2*(T12+t3))  # Fixed min/max calculation
                                    if nT != Int(2*T)
                                        continue
                                    end
                                    # check if MT is in the range of -T to T
                                    if abs(MT) > T
                                        continue
                                    end
                                    
                                    nch_count += 1
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    

    println("For J=",J, " T=",T," parity=",parity, " Number of channels: ", nch_count)
    open("channels.dat", "w") do io
        println(io, "For J=",J, " T=",T," parity=",parity, " Number of channels: ", nch_count)
    end
    # Now allocate arrays with the correct size
    α.nchmax = nch_count
    α.l = zeros(Int, nch_count)
    α.s12 = zeros(Float64, nch_count)
    α.J12 = zeros(Float64, nch_count)
    α.λ = zeros(Int, nch_count)
    α.s3 = zeros(Float64, nch_count)
    α.J3 = zeros(Float64, nch_count)
    α.J = zeros(Float64, nch_count)
    α.T12 = zeros(Float64, nch_count)
    α.T = zeros(Float64, nch_count)
    
    # Second pass to fill the channels
    ich = 0
    
    if nch_count > 0  # Only do second pass if we have channels
      open("channels.dat", "a") do io
        println(io, "---The coupling coefficients are")
        println(io, " a3b |( l ( s1 s2 ) s12 ) J12 ( λ s3 ) J3 ,   J; ( t1 t2 ) T12 , t3 , T >")
        for l in lmin:lmax
            for ns in Int(2*(s1-s2)):2:Int(2*(s1+s2))
                s12 = ns/2.0
                for nJ12 in Int(2*abs(l-s12)):2:Int(2*(l+s12))
                    J12 = nJ12/2.0
                    for λ in λmin:λmax
                        if (-1)^(l+λ) != parity
                            continue
                        end
                        for nJ3 in Int(2*abs(λ-s3)):2:Int(2*(λ+s3))
                            J3 = nJ3/2.0
                            for nJ in Int(2*abs(J12-J3)):2:Int(2*(J12+J3))
                                if nJ != Int(2*J)
                                    continue
                                end
                                
                                for nT12 in Int(2*abs(t1-t2)):2:Int(2*(t1+t2))
                                    T12 = nT12/2.0
                                    if (-1)^(l+s12+T12) !=-1
                                        continue
                                    end
                                    for nT in Int(2*abs(T12-t3)):2:Int(2*(T12+t3))
                                        if nT != Int(2*T)
                                            continue
                                        end
                                        # check if MT is in the range of -T to T
                                        if abs(MT) > T
                                            continue
                                        end
                                        ich += 1
                                        α.l[ich] = l
                                        α.s12[ich] = s12
                                        α.J12[ich] = J12
                                        α.λ[ich] = λ
                                        α.s3[ich] = s3
                                        α.J3[ich] = J3
                                        α.J[ich] = J
                                        α.T12[ich] = T12
                                        α.T[ich] = T
                                        print_channel_info(io, ich, l, s1, s2, s12, J12, λ, s3, J3, J, T12, t3, T)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
      end # end open
    end # end if 
    

    return α
end





# Function to update parameters
function update_parameters(params)
    global lmax = params["lmax"]
    global lmin = params["lmin"]
    global t1 = params["t1"]
    global t2 = params["t2"]
    global t3 = params["t3"]
    global s1 = params["s1"]
    global s2 = params["s2"]
    global s3 = params["s3"]
    global λmin = params["λmin"]
    global λmax = params["λmax"]
    global Jmin = params["Jmin"]
    global Jmax = params["Jmax"]
    global Tmin = params["Tmin"]
    global Tmax = params["Tmax"]
    global MT = params["MT"]
end

function print_channel_info(io, ich, l, s1, s2, s12, J12, λ, s3, J3, J, T12, t3, T)
    @printf(io, "%4d |(%2d (%2.1f %2.1f) %2.1f) %4.1f (%2d %2.1f) %3.1f, %3.1f; (%2.1f %2.1f) %2.1f, %2.1f, %2.1f > \n",
            ich, l, s1, s2, s12, J12, λ, s3, J3, J, t1, t2, T12, t3, T)
end

end # module