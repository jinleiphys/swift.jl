# Goal
把 Lazauskas-Carbonell PRC 84,034002 (2011) Eq.17 的 Green 定理 n-d 弹性散射振幅,正确映射到 swift.jl 的算符上,使收敛后的结果命中 benchmark (Re δ=105.49°, η=0.4649),而不是现在的 δ≈0、η≈0.9、|f|×12≈4π 偏小。

# Current state
- swift.jl = R-space Faddeev,Lagrange-Laguerre 基,约化表象 F(x,y)/(xy),复标度(back-rotation,实基组,旋转放在 V/T/源里)。
- 物理 Jacobi 坐标(非质量标度):x 动能系数 ħ²/m(μ_x=m/2),y 动能系数 (3/4)ħ²/m(μ_y=2m/3)。ħ²/m=41.471。
- V 矩阵 = V_x ⊗ Ny(x 积分 + y 重叠度规都含在 V 里)。Rxy=Rxy_31+Rxy_32,识别为 P⁻+P⁺;identical fermion 下 Rxy_32=Rxy_31ᵀ,Rxy 对称,Rxy=2Rxy_31。
- 散射解:[E·B−T−V·(I+Rxy)]ψ_sc = 2·V·Rxy_31·φ (=V·Rxy·φ),GMRES,rel_res~2e-5。ψ_total=ψ_in+ψ_sc 是单个 Faddeev 分量。Ψ̄=(I+Rxy)ψ_total。
- ψ_in = compute_initial_state_vector:φ_d(x)·F_λ(qy e^{iθ})/(ϕx ϕy),F_λ=COULCC Riccati 正则(随 CS e^{+qy sinθ} 增长)。
- bra 我搭成 φ_d(x)·ĥ⁻_λ(qy e^{iθ})/(ϕxϕy),ĥ⁻=gc−i·fc。
- deuteron c-norm C_n=transpose(evec)·Ix·evec·e^{iθ}(约化 1D x),Ix=单位+(−1)^(i−j)/√(xx_i xx_j) 是正则 Lagrange 重叠度规。CS 振幅必须用双线性 c-product(transpose 非 dot,bra 不取共轭)。
- 标量 2-body 已锁定(test_2body_cs_1S0.jl,MT ¹S₀ δ=63.512°,η=1,与 COLOSS scatt.f 一致):f=−(2μ/ħ²k²)e^{iθ}⟨F|V|u⟩=−(1/E)e^{iθ}M,S=1+2ik·f,正则 F bra,每径向积分一个 e^{iθ},V 在 stiffness 表象无需额外 B 度规。
- COLOSS(2-body 参考,/Users/jinlei/Desktop/code/COLOSS/src/scatt.f):f_sc=Σ dn²/(E−Eₙ),dn=⟨V·fc_rotated|eigvec⟩,V 乘 fc_rotated 保护增长的 bra;f_born=∫rw·V·fc² 未旋转。

# 原文 Eq.17 (中性 n+d) — 目标公式
f_nm(ŷ) = −C_n⁻¹ (m/ħ²) ∬ φ_n*(x_i e^{−iθ}) · [e^{−iq_n y_i e^{iθ}}/|y_i|] · [V_j(x_j e^{iθ})+V_k(x_k e^{iθ})] · Ψ̄_m(x_i,y_i) · e^{6iθ} d³x_i d³y_i
C_n = ∫ φ_n*(x_i e^{−iθ}) φ_n(x_i e^{iθ}) e^{3iθ} d³x_i
- i = 入射分区(deuteron 对 = pair jk;投射粒子 i);V_j,V_k = 另两个对(含投射粒子 i)的势,即 V₂₃+V₃₁(排除 deuteron 对 V_i)。
- Ψ̄ = 完整三体波函数(不是 ψ_sc)。bra y 核 e^{−iqy e^{iθ}}/|y| = 出射自由 Green 核(l=0 无库仑=e^{−iqy})。
- 前因子 −C_n⁻¹(m/ħ²)e^{6iθ},量级完全定死无自由标度(真判据,不许凑数)。
- 收敛保护机制:bra 里 deuteron φ_n(x_i) 截断 x_i + [V_j+V_k] 短程截断 x_j → 几何上 y_i 被一起截断。

# Tried (with results)
- T1. M = bra^T·(Rxy·V·Ψ̄)(Hankel bra,前因子 −C_n⁻¹ m/ħ² e^{6iθ})。结果:|M| 随网格暴涨 674→968→1645,发散。诊断:V 作用在 Ψ̄ 侧、Rxy 在后,V 没紧贴增长的 bra。[status: failed]
- T2. 同上但 ψ_sc-only(Ψ̄_sc=(I+Rxy)ψ_sc):|M_sc| 51→62→17 乱跳,仍不收敛。[status: failed]
- T3. 早期 V·Rxy·ψ_total + 正则 F bra + 前因子 −1/E_cm:η=0.471≈benchmark 0.4649 但 δ=46.78°(非 105.49)。后判定为 scale+phase 巧合(扫一个实标度+整数 e^{inθ} 必能命中任意单个复数 f_target,不算验证)。[status: failed — 巧合非验证]
- T4. 用户指正(WS/短程 V + back-rotation 本该自动截断)后,P 算符代数推:⟨bra|V₂₃+V₃₁|Ψ̄⟩=⟨(P⁻+P⁺)bra|V₁₂|Ψ̄⟩=(Rxy·bra)ᵀ·V·Ψ̄(先 rearrange bra,再让 V 乘上保护)。改成 Mp=(Rxy·bra)ᵀ·V·Ψ̄:|Mp| 收敛 3.71→4.36→4.37(✓ 收敛修好)。但物理错:|f|≈0.11(×12≈4π 偏小)、δ≈0°、η≈0.9(几乎无散射)。[status: partial — 收敛对,物理错]
- T5. 在 T4 收敛 ordering 下扫 bra type(ĥ⁻=gc−ifc / ĥ⁺=gc+ifc / 正则 F):三者 |Mp| 几乎相同(被 irregular G 主导),δ 只微动(±2°),η~0.9-1.1。bra type 不是杠杆。[status: failed]
- 关键矛盾:T1 与 T4 两种 ordering 数值差 674 vs 4.4 → 截断基里 Rxy 不严格对称,"Rxy^T=Rxy" 的代数步骤崩了。δ≈0 说明 ψ_sc 的散射信息没进到 Mp(Born/ψ_in 项主导 → 近实数)。

# 三个待 PK 的核心子问题
(a) [V_j+V_k]Ψ̄ 到底对应 swift 的哪个算符组合?(Rxy·bra)ᵀ·V·Ψ̄ 对不对?要不要分 Rxy_31/Rxy_32 不同方向?V 在哪一侧?Ψ̄=(I+Rxy)ψ_total 还是别的?为什么 δ≈0(散射信息丢了)?
(b) 前因子 ×12≈4π 偏小:缺 4π 角向归一化?质量该用 2μ_y/ħ²(=4/3·m/ħ²)而非 m/ħ²?C_n 该用 3D e^{3iθ} 而非约化 e^{iθ}?e^{6iθ} Jacobian 在约化表象该是别的幂(可能 e^{2iθ})?
(c) bra y 核 e^{−iqy e^{iθ}}/|y| 在约化 F/(xy) 表象里到底是 ĥ⁻_λ 还是 ĥ⁻_λ/y?irregular G 主导 |Mp| 是否本身就是 bug 信号(正则解应当只取出射分量,G 不该主导)?

# Budget / constraints
- 不许凑数:前因子量级定死,合格判据 = 收敛(mesh + θ-plateau)且命中 benchmark,不是扫标度命中单点。
- 必须留在 swift.jl 框架(Lagrange-Laguerre 约化表象、现有 V/Rxy/T 算符、GMRES 散射解、back-rotation CS)。
- 已锁定不动:束缚态、2-body 标量约定(test_2body_cs_1S0.jl)、bound2b 归一化。
- 计算:本地 Julia smoke test 即可(nx~12-20, ny~30-50),分钟级;weeks-not-months。
- 用户明确要走"路 B"(死磕 3 体 Eq.17 → Rxy 映射),不要"路 A"(把 elastic 当 y 通道有效 2-body 套 COLOSS)。
- 文件:swift/test_eq17_green.jl, swift/scattering.jl, swift/matrices_optimized.jl, TODO.md, memory/cs-amplitude-cproduct-cnorm.md。原文 ~/Downloads/PhysRevC.84.034002.pdf。COLOSS /Users/jinlei/Desktop/code/COLOSS/src/scatt.f。
