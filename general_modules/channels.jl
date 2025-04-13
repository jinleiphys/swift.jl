module channels 

   # channel parameters
   lmax = 0 # maximum l
   lmin = 0 # minimum l

   s12 = 0.0 # coupling of s1 and s2
   J12 = 0.0 # coupling of s12 and l

   λmax=0  # maximum λ  
   λmin=0  # minimum λ
   J3=0.0 # coupling of λ and s3 

   J=0.0 # coupling of J12 and J3
   s1=0.0  # spin of particle 1 
   s2=0.0  # spin of particle 2
   s3=0.0  # spin of particle 3

   # for bound state J=Jmin=Jmax 
   Jmin=0.0 # minimum J
   Jmax=0.0 # maximum J

   t1=0.0 # isospin of particle 1
   t2=0.0 # isospin of particle 2
   t3=0.0 # isospin of particle 3
   T12=0.0 # coupling of t1 and t2
   T=0.0 # coupling of T12 and t3
   Tmin=0.0 # minimum T 
   Tmax=0.0 # maximum T
   MT=0.0 # third component of T, fix for the given system by charge conservation


   mutable struct α_bar  # for the channel coupling 
    αin=[] 
    αout=[]
   end 

   mutable struct nch3b # channel index for the three body coupling 
    nchmax =0 # maximum number of channels
    l=[]
    s12=[]
    J12=[]
    λ=[]
    J3=[]
    J=[]
    T12=[]
    T=[]
   end 

   #Initialize the structures
   αbar = α_bar()
   α=nch3b()

   # channel index of the three body channels 
   #|(l_{12} (s_1 s_2) s_{12}) J_{12}, (\lambda_3 s_3) J_3, J; (t_1 t_2) T_{12}, t_3, T M_T\rangle.
   function α3b()
    α.nchmax=0
    for l in lmin:lmax
        for s12 in 0:2*s1+1
            for J12 in 0:2*s12+1
                for λ in λmin:λmax
                    for J3 in 0:2*λ+1
                        for J in max(abs(J12-J3),abs(l-J12)):min(J12+J3,l+J12)
                            for T12 in 0:2*t1+1
                                for T in max(abs(T12-t3),abs(t2-T12)):min(T12+t3,t1+T12)
                                    α.nchmax+=1
                                end 
                            end 
                        end 
                    end 
                end 
            end 
        end




   end  

   




end module 