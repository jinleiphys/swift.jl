
## Faddeev Equations
In this discussion, I examine the application of the Faddeev equations to a system of three identical particles to determine the bound state energy. This approach assumes that the particles are indistinguishable in terms of their pairwise interactions, as exemplified by systems like the triton (composed of two neutrons and one proton) and the $^{12}$C nucleus (modeled as three alpha particles). However, the nucleon-nucleon (NN) force introduces a distinguishing feature through the third component of isospin, which breaks the symmetry in certain interactions, necessitating careful consideration in the formulation of the equations and the resulting energy calculations. When a two-body bound state exists within this system, the three-body state energy must satisfy the condition $E < E_{2b}$, where $E_{2b}$ represents the energy of the two-body ground state. This inequality ensures that the three-body system remains stable relative to its two-body subsystem.

To streamline the analysis, the three identical particles are labeled as 1, 2, and 3. The total wave function $|\Psi\rangle$ of this system is governed by the Schrödinger equation:
$$
H |\Psi\rangle = E |\Psi\rangle = (H_0 + V_{12} + V_{23} + V_{31} + V_{123}) |\Psi\rangle,
$$
where $H_0$ is the free Hamiltonian describing the kinetic energy of the particles, $V_{ij}$ represents the two-body interaction potential between particles $i$ and $j$, and $V_{123}$ denotes the three-body interaction potential. This equation encapsulates the dynamics of the system, balancing the free motion and the interactions among the particles.

A key feature of quantum mechanics is the indistinguishability of identical particles, meaning they lose their individual identities. As a result, the physical state of the system must remain unchanged under the exchange of any two particles, up to a phase factor. Mathematically, applying the permutation operator $P_{12}$ (which swaps particles 1 and 2) twice yields:
$$
P_{12}P_{12} |\Psi(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3)\rangle = P_{12} \epsilon |\Psi(\mathbf{r}_2, \mathbf{r}_1, \mathbf{r}_3)\rangle = \epsilon^2 |\Psi(\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3)\rangle.
$$
Since $P_{12}^2 = 1$ (the identity operator), it follows that $\epsilon^2 = 1$, so $\epsilon = \pm 1$. Thus, the wave function $|\Psi\rangle$ is either symmetric ($\epsilon = +1$) or antisymmetric ($\epsilon = -1$) under particle exchange. Particles with symmetric wave functions obey Bose-Einstein statistics and are classified as bosons, while those with antisymmetric wave functions obey Fermi-Dirac statistics and are known as fermions. Relativistic quantum mechanics further reveals that a particle’s spin determines its statistics: particles with integer spin (e.g., 0, 1) are bosons, while those with half-integer spin (e.g., 1/2, 3/2) are fermions. In the case under consideration, the particles are fermions with half-integer spin, consistent with the properties of nucleons like protons and neutrons in the triton.

To understand the role of permutation operators in the interactions, consider the two-body potential matrix element $V_{12}$:
$$
\langle \mathbf{a}'\mathbf{b}'\mathbf{c}' | V_{12} | \mathbf{a}\mathbf{b}\mathbf{c} \rangle = \delta(\mathbf{c} - \mathbf{c}') V(\mathbf{a}'\mathbf{b}', \mathbf{a}\mathbf{b}),
$$
where $\mathbf{a}, \mathbf{b}, \mathbf{c}$ denote the coordinates of particles 1, 2, and 3, respectively, and the delta function $\delta(\mathbf{c} - \mathbf{c}')$ reflects that $V_{12}$ acts only on particles 1 and 2, leaving particle 3 unaffected. Applying permutation operators, such as $P_{12}P_{23}V_{12}P_{23}P_{12}$, transforms the potential:
$$
\begin{align}
\langle \mathbf{a}'\mathbf{b}'\mathbf{c}' | P_{12}P_{23}V_{12}P_{23}P_{12} | \mathbf{a}\mathbf{b}\mathbf{c} \rangle &= \langle \mathbf{b}'\mathbf{a}'\mathbf{c}' | P_{23}V_{12}P_{23} | \mathbf{b}\mathbf{a}\mathbf{c} \rangle \\
&= \langle \mathbf{b}'\mathbf{c}'\mathbf{a}' | V_{12} | \mathbf{b}\mathbf{c}\mathbf{a} \rangle \\
&= \delta(\mathbf{a} - \mathbf{a}') V(\mathbf{b}'\mathbf{c}', \mathbf{b}\mathbf{c}) \\
&= \langle \mathbf{a}'\mathbf{b}'\mathbf{c}' | V_{23} | \mathbf{a}\mathbf{b}\mathbf{c} \rangle.
\end{align}
$$
This demonstrates that:
$$
V_{23} = P_{12}P_{23} V_{12} P_{23}P_{12},
$$
showing how the interaction between particles 2 and 3 can be expressed in terms of the interaction between particles 1 and 2 through permutations. Similarly, one can derive:
$$
V_{31} = P_{23}P_{12} V_{12} P_{12}P_{23}.
$$
For the three-body interaction $V_{123}$, symmetry under particle exchange is assumed, satisfying $[V_{123}, P_{ij}] = 0$. It is often expressed as a sum over permuted terms:
$$
\begin{align}
V_{123} & = V_{12}^{(3)} +  V_{23}^{(1)}  +  V_{31}^{(2)} , \\
& = V_{12}^{(3)} + P_{12}P_{23} V_{12}^{(3)}P_{23} P_{12} + P_{23}P_{12} V_{12}^{(3)} P_{12}P_{23}, \\
\end{align}
$$
where $V_{12}^{(3)}$ is symmetric with respect to $P_{12}$, ensuring the total three-body potential respects the indistinguishability of the particles.

The Faddeev approach decomposes the total wave function $|\Psi\rangle$ into components that isolate the contributions of each two-body interaction. These components are defined as:
$$
\begin{align}
|\psi_{1}\rangle &= G_0 ( V_{23}+V_{123}^{(1)}) |\Psi\rangle, \\
|\psi_{2}\rangle &= G_0 ( V_{31} +V_{123}^{(2)} ) |\Psi\rangle, \\
|\psi_{3}\rangle &= G_0 ( V_{12}  +V_{123}^{(3)} )|\Psi\rangle, \\
\end{align}
$$
where $G_0 = 1/(E - H_0)$ is the free Green’s function, describing the propagation of the system in the absence of interactions, and the total wave function is the sum:
$$
|\Psi\rangle = |\psi_{1}\rangle + |\psi_{2}\rangle + |\psi_{3}\rangle.
$$
This decomposition simplifies the three-body problem by breaking it into coupled equations for each Faddeev component, which can be solved iteratively or numerically to determine the bound state energy $E$. 

## Jacobi coordinates and partial wave decomposition 

By expressing these equations in **Jacobi coordinates**, we can naturally separate the center-of-mass motion from the relative dynamics, simplifying the problem. For three particles with positions $\mathbf{r}_1, \mathbf{r}_2, \mathbf{r}_3$, the Jacobi coordinates for the pair (2,3) with particle 1 as the spectator are defined as:
$$
\mathbf{x}_1 = \mathbf{r}_2 - \mathbf{r}_3, \quad \mathbf{y}_1 = \frac{1}{2}(\mathbf{r}_2 + \mathbf{r}_3) - \mathbf{r}_1,
$$
where $\mathbf{x}_1$ is the relative coordinate between particles 2 and 3, and $\mathbf{y}_1$ describes the position of particle 1 relative to the center of mass of the pair (2,3). Similar coordinates are defined cyclically for the other pairs:
$$
\mathbf{x}_2 = \mathbf{r}_3 - \mathbf{r}_1, \quad \mathbf{y}_2 = \frac{1}{2}(\mathbf{r}_3 + \mathbf{r}_1) - \mathbf{r}_2,
$$
$$
\mathbf{x}_3 = \mathbf{r}_1 - \mathbf{r}_2, \quad \mathbf{y}_3 = \frac{1}{2}(\mathbf{r}_1 + \mathbf{r}_2) - \mathbf{r}_3.
$$
These coordinates ensure that the center-of-mass motion can be factored out, allowing us to focus on the relative motion of the particles.

The total wave function $\Psi(\mathbf{x}_i, \mathbf{y}_i)$ of the three-body system is decomposed into three Faddeev components, each associated with a specific pair of interacting particles:
$$
\Psi(\mathbf{x}_i, \mathbf{y}_i) = \psi_1(\mathbf{x}_1, \mathbf{y}_1) + \psi_2(\mathbf{x}_2, \mathbf{y}_2) + \psi_3(\mathbf{x}_3, \mathbf{y}_3),
$$
where $\psi_i(\mathbf{x}_i, \mathbf{y}_i)$ represents the Faddeev component corresponding to the interaction of the pair opposite to particle $i$. For identical particles, these components are related by symmetry operations due to their indistinguishability.

To account for this symmetry, we introduce permutation operators that describe transitions between the Faddeev components. Define the **positive permutation operator** as:
$$
P^+ = P_{12}P_{23},
$$
where $P_{ij}$ swaps the labels of particles $i$ and $j$. (Note: this corresponds to the cyclic permutation $1 \to 2 \to 3 \to 1$ of particle labels; the opposite convention $P^+ = P_{23}P_{12}$ appears in some references and just exchanges the roles of $P^+$ and $P^-$ below.) This operator cyclically increases the component index:
$$
\begin{align}
\psi_2(\mathbf{x}_2, \mathbf{y}_2) &= P^+ \psi_1(\mathbf{x}_1, \mathbf{y}_1), \\
\psi_3(\mathbf{x}_3, \mathbf{y}_3) &= P^+ \psi_2(\mathbf{x}_2, \mathbf{y}_2), \\
\psi_1(\mathbf{x}_1, \mathbf{y}_1) &= P^+ \psi_3(\mathbf{x}_3, \mathbf{y}_3).
\end{align}
$$
The action of $P^+$ corresponds to a cyclic permutation $(1 \to 2 \to 3 \to 1)$. The **inverse permutation operator** is:
$$
P^- = (P^+)^{-1} = P_{23}P_{12},
$$
which decreases the component index cyclically:
$$
\begin{align}
\psi_3(\mathbf{x}_3, \mathbf{y}_3) &= P^- \psi_1(\mathbf{x}_1, \mathbf{y}_1), \\
\psi_1(\mathbf{x}_1, \mathbf{y}_1) &= P^- \psi_2(\mathbf{x}_2, \mathbf{y}_2), \\
\psi_2(\mathbf{x}_2, \mathbf{y}_2) &= P^- \psi_3(\mathbf{x}_3, \mathbf{y}_3).
\end{align}
$$
These operators exploit the symmetry of identical particles, ensuring that the Faddeev components transform appropriately under particle exchanges. For fermions (as in the case of nucleons with half-integer spin), the wave function must be antisymmetric under the exchange of any two particles, which imposes additional constraints on the components.

In coordinate space, the Faddeev equations for the three components are formulated to account for the interactions within each pair and the three-body force. They can be written as:
$$
\begin{align}
(E - H_0 - V_{23} - V_{123}^{(1)}) \psi_1(\mathbf{x}_1, \mathbf{y}_1) &= (V_{23} + V_{123}^{(1)}) [\psi_2(\mathbf{x}_2, \mathbf{y}_2) + \psi_3(\mathbf{x}_3, \mathbf{y}_3)], \\
(E - H_0 - V_{31} - V_{123}^{(2)}) \psi_2(\mathbf{x}_2, \mathbf{y}_2) &= (V_{31} + V_{123}^{(2)}) [\psi_3(\mathbf{x}_3, \mathbf{y}_3) + \psi_1(\mathbf{x}_1, \mathbf{y}_1)], \\
(E - H_0 - V_{12} - V_{123}^{(3)}) \psi_3(\mathbf{x}_3, \mathbf{y}_3) &= (V_{12} + V_{123}^{(3)}) [\psi_1(\mathbf{x}_1, \mathbf{y}_1) + \psi_2(\mathbf{x}_2, \mathbf{y}_2)],
\end{align}
$$
where:
- $E$ is the total energy of the three-body system,
- $H_0$ is the free Hamiltonian for the relative motion in Jacobi coordinates,
- $V_{ij}$ is the two-body potential acting between particles $i$ and $j$, expressed in the appropriate Jacobi coordinates,
- $V_{123}^{(i)}$ represents the component of the three-body interaction associated with the Faddeev component $\psi_i$, ensuring symmetry under particle permutations.

Each equation describes the dynamics of one Faddeev component, with the left-hand side representing the free motion and interactions within a specific pair, and the right-hand side coupling it to the other two components through their respective interactions.

For identical particles, the symmetry of the system allows us to simplify the problem significantly. Since the particles are indistinguishable, the Faddeev components are related by the permutation operators $P^+$ and $P^-$. This symmetry implies that solving for one Faddeev component is sufficient to determine the entire wave function. Assuming the particles are identical and the interactions are symmetric, we can focus on a single component, say $\psi_1$, and express the other components as $\psi_2 = P^+ \psi_1$ and $\psi_3 = P^+ \psi_2 = (P^+)^2 \psi_1$. Substituting these into the first Faddeev equation and using the permutation properties, we obtain a single equation:
$$
(E - H_0 - V) \psi_1 = V (P^+ + P^-) \psi_1,
$$
where $V = V_{23} + V_{123}^{(1)}$ represents the total interaction potential (two-body and three-body contributions) acting in the coordinate system of $\psi_1$. The operator $P^+ + P^-$ accounts for the contributions of the other Faddeev components $\psi_2$ and $\psi_3$ through their permutation relationships. In operator form, this can be written as:
$$
(E - H_0 - V) |\psi\rangle = V (P^+ + P^-) |\psi\rangle,
$$
where $|\psi\rangle$ denotes the Faddeev component in the abstract state space.


Now, let's define the angular momentum basis. The starting point is the naturally antisymmetrized basis of two nucleons, denoted $|\beta\rangle$:
$$
|\beta_{12}\rangle = | l_{12} (s_1 s_2) s_{12} J_{12}; (t_1 t_2) T_{12} \rangle,
$$
where $l_{12}$, $s_{12}$, and $T_{12}$ are restricted by the antisymmetrization condition $(-)^{l_{12} + s_{12} + T_{12}} = -1$. Please note that for the identical particles, the NN force is different for the third component of $T_{12}$. The third nucleon, with all possible allowable quantum numbers, is coupled to $|\beta\rangle$ to form all possible three-body states $|\alpha_3\rangle$:
$$
|\alpha_3\rangle = |(l_{12} (s_1 s_2) s_{12}) J_{12}, (\lambda_3 s_3) J_3, J; (t_1 t_2) T_{12}, t_3, T M_T\rangle.
$$
Then, in the angular momentum basis, one has the following relations:
$$
\langle x_3 y_3 \alpha_3 | x_3' y_3' \alpha_3' \rangle = \frac{\delta(x_3 - x_3')}{x_3^2} \frac{\delta(y_3 - y_3')}{y_3^2} \delta_{\alpha_3, \alpha_3'},
$$
and
$$
1 = \sum_{\alpha_3} \int x_3^2 y_3^2 \, dx_3 \, dy_3 \, | x_3 y_3 \alpha_3 \rangle \langle x_3 y_3 \alpha_3 |.
$$
The kinetic energy operator $H_0$ in the Jacobi coordinates $x_3$, $y_3$, $\alpha_3$ is given by:
$$
H_0 = -\Delta^{\alpha_3}(x_3, y_3) = \frac{\hbar^2}{2\mu_{12}} \left( -\frac{\partial^2}{\partial x_3^2} + \frac{l_{12}(l_{12} + 1)}{x_3^2} \right) + \frac{\hbar^2}{2\mu_{3}} \left( -\frac{\partial^2}{\partial y_3^2} + \frac{\lambda_{3}(\lambda_{3} + 1)}{y_3^2} \right).
$$
The pair potential in the angular momentum basis is:
$$
\langle x_3 y_3 \alpha_3 | V_{12} | x_3' y_3' \alpha_3' \rangle = \frac{\delta(y_3 - y_3')}{y_3^2} V_{\alpha_3, \alpha_3'}(x_3, x_3') \delta_{J_{12}, J_{12}'} \delta_{s_{12}, s_{12}'},
$$
which simplifies to the following form if the potential is local:
$$
\langle x_3 y_3 \alpha_3 | V_{12} | x_3' y_3' \alpha_3' \rangle = \frac{\delta(x_3 - x_3')}{x_3^2} \frac{\delta(y_3 - y_3')}{y_3^2} V_{\alpha_3, \alpha_3'}(x_3) \delta_{J_{12}, J_{12}'} \delta_{s_{12}, s_{12}'}.
$$
## Transformation of Jacobi coordinates 
![Jacobi](pic/Jacobi.001.png)

In the context of Jacobi coordinates, the transformation between different set of coordinates is given by linear transformations. For a three-body system with equal masses, and the total mass $M=3m$, the following coordinate transformations can be defined:   

For the transformation between the coordinate associate with third particle of 3 and 1, the relationship is given by 
$$
\begin{bmatrix}
\vec{x}_1 \\
\vec{y}_1
\end{bmatrix}
= 
\begin{bmatrix}
-\frac{1}{2} & 1 \\
-\frac{3}{4} & -\frac{1}{2}
\end{bmatrix}
\begin{bmatrix}
\vec{x}_3 \\
\vec{y}_3
\end{bmatrix}
$$
and similarly, the transformation between the coordinate associate with third particle of 3 and 2, the relationship is given by 
$$
\begin{bmatrix}
\vec{x}_2 \\
\vec{y}_2
\end{bmatrix}
= 
\begin{bmatrix}
-\frac{1}{2} & -1 \\
\frac{3}{4} & -\frac{1}{2}
\end{bmatrix}
\begin{bmatrix}
\vec{x}_3 \\
\vec{y}_3
\end{bmatrix}
$$
## Evaluation of Matrix Elements

First, let us consider the potential $V_{\alpha_3, \alpha_3'}(x_3, x_3')$. The nucleon-nucleon (NN) force depends on the third component of the pair's isospin. The potential $V_{12}$ acts only on particles 1 and 2, so it is diagonal in the coordinate and isospin of particle 3, and it depends on the relative coordinate $x_3$ and the isospin projections of particles 1 and 2. The NN potential $V_{12}$ depends on the isospin projections $m_{t_1}$ and $m_{t_2}$, and it is typically expressed in the total isospin basis of the pair (1,2). To compute this term exactly, one must decouple the isospin state. For this, we have:
$$
\begin{align}
V_{\alpha_3, \alpha_3'}(x_3, x_3') &= \langle x_3 (t_1 t_2)T_{12} t_3; T M_T | V_{12} | x_3' (t_1 t_2)T_{12}' t_3; T' M_T \rangle \\
&= \sum_{m_{t_{12}} m'_{t_{12}}} \sum_{m_{t_3} m'_{t_3}} \langle (t_1 t_2)T_{12} t_3; T M_T | T_{12} m_{t_{12}} t_3 m_{t_3} \rangle \langle x_3 T_{12} m_{t_{12}} t_3 m_{t_3} | V_{12} | x_3' T_{12}' m'_{t_{12}} t_3 m'_{t_3} \rangle \\
& \quad \times \langle T_{12}' m'_{t_{12}} t_3 m'_{t_3} | (t_1 t_2)T_{12}' t_3; T' M_T \rangle \\
&= \sum_{m_{t_{12}}} \langle T_{12} m_{t_{12}} t_3 (M_T - m_{t_{12}}) | T M_T \rangle 
\langle T_{12} m_{t_{12}} t_3 (M_T - m_{t_{12}}) | T' M_T \rangle  V_{12}^{T_{12}, m_{t_{12}}}(x_3, x_3') \\
& \times \delta_{\lambda_3,\lambda_3'}\delta_{J_3,J_3'}\delta_{s_{12},s_{12}'}\delta_{T_{12}, T_{12}'}\delta_{J_{12},J_{12}'},
\end{align}
$$
where the Kronecker delta $\delta_{T_{12}, T_{12}'}$ ensures that the potential is diagonal in the pair isospin. It should be noted that in neutron-deuteron (nd) or proton-deuteron (pd) scattering, the total isospin $T$ is not strictly conserved due to charge independence breaking (CIB) in the nucleon-nucleon (NN) interaction. Isospin symmetry is an approximate concept under which protons and neutrons are treated as two states of the same particle (the nucleon), with the strong nuclear force acting identically regardless of charge state. However, real NN forces exhibit small but measurable differences between neutron-neutron (nn), proton-proton (pp), and neutron-proton (np) interactions, even after neglecting the Coulomb force.

In three-nucleon (3N) systems, CIB introduces off-diagonal terms in the two-body t-matrix elements that couple different total isospin states. For nd scattering with projection $M_T = -1/2$, the t-matrix in isospin space includes terms such as:
$$
\langle (1 \frac{1}{2}) \frac{1}{2} -\frac{1}{2} | V_{12} | (1 \frac{1}{2}) \frac{3}{2} -\frac{1}{2} \rangle = \frac{1}{3} \sqrt{2} (\tilde{V}_{nn}^{t=1} - \tilde{V}_{np}^{t=1}),
$$
where $\tilde{V}_{nn}^{t=1}$ and $\tilde{V}_{np}^{t=1}$ are the nn and np t-matrices, respectively. This non-zero off-diagonal element drives an admixture of the $T = 3/2$ state into the dominant $T = 1/2$ state (the isospin of the deuteron ground state), with the mixing amplitude reaching up to approximately 10% in certain channels.


Now, let us consider the matrix element $\langle x_3 y_3 \alpha_3 | V_{12} | P^+ \psi \rangle$. The evaluation can be expanded as follows:
$$
\begin{align}
\langle x_3 y_3 \alpha_3 | V_{12} | P^+ \psi \rangle &= \sum_{\alpha_3'} \int x_3'^2 \, dx_3' \, \langle x_3 y_3 \alpha_3 | V_{12} | x_3' y_3 \alpha_3' \rangle \langle x_3' y_3 \alpha_3' | P^+ \psi \rangle \\
&= \sum_{\alpha_3'} \int x_3'^2 \, dx_3' \, V_{\alpha_3, \alpha_3'}(x_3, x_3') \langle x_3' y_3 \alpha_3' | P^+ \psi \rangle \delta_{J_{12}, J_{12}'} \delta_{s_{12}, s_{12}'}
\end{align}
$$

For the term $\langle x_3' y_3 \alpha_3' | P^+ \psi \rangle$, one can compute it as follows:
$$
\begin{align}
\langle x_3' y_3 \alpha_3' | P^+ \psi \rangle &= \sum_{\alpha_1} \int x_1^2 y_1^2 \, dx_1 \, dy_1 \, \langle x_3' y_3 \alpha_3' | x_1 y_1 \alpha_1 \rangle \langle x_1 y_1 \alpha_1 | P^+ \psi \rangle \\
&= \sum_{\alpha_1} \int x_1^2 y_1^2 \, dx_1 \, dy_1 \int_{-1}^1 d\cos\theta \, \mathcal{G}_{\alpha_3', \alpha_1}(x_3', y_3, \cos\theta) \frac{\delta(x_1 - \pi_1(x_3', y_3, \cos\theta))}{x_1^2} \\
&\quad \times \frac{\delta(y_1 - \xi_1(x_3', y_3, \cos\theta))}{y_1^2} \langle x_1 y_1 \alpha_1 | \psi \rangle
\end{align}
$$
where $\pi_1$ and $\xi_1$ are defined as:
$$
\pi_1 = \sqrt{\frac{1}{4} x_3^2 - x_3 y_3 \cos\theta + y_3^2}
$$
and
$$
\xi_1 = \sqrt{\frac{9}{16} x_3^2 + \frac{3}{4} x_3 y_3 \cos\theta + \frac{1}{4} y_3^2}
$$

Thus, the term $\langle x_3' y_3 \alpha_3' | P^+ \psi \rangle$ can be simplified as:
$$
\langle x_3' y_3 \alpha_3' | P^+ \psi \rangle = \sum_{\alpha_1} \int_{-1}^1 d\cos\theta \, \mathcal{G}_{\alpha_3', \alpha_1}(x_3', y_3, \cos\theta) \langle \pi_1 \xi_1 \alpha_1 | \psi \rangle
$$
where $\mathcal{G}_{\alpha_3', \alpha_1}(x_3', y_3, \cos\theta)$ is given by:
$$
\begin{align}
\mathcal{G}_{\alpha_3', \alpha_1}(x_3', y_3, \cos\theta) &= \sum_{L S} (2S + 1) \sqrt{(2 J_{12} + 1)(2 J_3 + 1)(2 J_{23} + 1)(2 J_1 + 1)} \begin{Bmatrix}
l_{12} & s_{12} & J_{12} \\
\lambda_3 & s_3 & J_3 \\
L & S & J
\end{Bmatrix} \\
&\quad \times \begin{Bmatrix}
l_{23} & s_{23} & J_{23} \\
\lambda_1 & s_1 & J_1 \\
L & S & J
\end{Bmatrix} 8 \pi^2 \sum_{M = -L}^L \left\{ Y_{l_{12}}^{m_{l_{12}} *}(\hat{x}_3) Y_{\lambda_3}^{m_{\lambda_3} *}(\hat{y}_3) \right\}^{L M_L} \left\{ Y_{l_{23}}^{m_{l_{23}}}(\hat{x}_1) Y_{\lambda_1}^{m_{\lambda_1}}(\hat{y}_1) \right\}^{L M_L} \\
&\quad \times (-)^{s_{23} + 2 s_1 + s_2 + s_3} \sqrt{(2 s_{12} + 1)(2 s_{23} + 1)} \begin{Bmatrix}
s_1 & s_2 & s_{12} \\
s_3 & S & s_{23}
\end{Bmatrix} \\
&\quad \times (-)^{T_{23} + 2 t_1 + t_2 + t_3} \sqrt{(2 T_{12} + 1)(2 T_{23} + 1)} \begin{Bmatrix}
t_1 & t_2 & T_{12} \\
t_3 & T & T_{23}
\end{Bmatrix}
\end{align}
$$

Similarly, one has 
$$
\begin{align}
\mathcal{G}_{\alpha_3', \alpha_2}(x_3', y_3, \cos\theta) &= \sum_{L S} (2S + 1) \sqrt{(2 J_{12} + 1)(2 J_3 + 1)(2 J_{31} + 1)(2 J_2 + 1)} \begin{Bmatrix}
l_{12} & s_{12} & J_{12} \\
\lambda_3 & s_3 & J_3 \\
L & S & J
\end{Bmatrix} \\
&\quad \times \begin{Bmatrix}
l_{31} & s_{31} & J_{31} \\
\lambda_2 & s_2 & J_2 \\
L & S & J
\end{Bmatrix} 8 \pi^2 \sum_{M = -L}^L \left\{ Y_{l_{12}}^{m_{l_{12}} *}(\hat{x}_3) Y_{\lambda_3}^{m_{\lambda_3} *}(\hat{y}_3) \right\}^{L M_L} \left\{ Y_{l_{31}}^{m_{l_{31}}}(\hat{x}_2) Y_{\lambda_2}^{m_{\lambda_2}}(\hat{y}_2) \right\}^{L M_L} \\
&\quad \times (-)^{s_{12} + 2 s_3 + s_1 + s_2} \sqrt{(2 s_{12} + 1)(2 s_{31} + 1)} \begin{Bmatrix}
s_3 & s_1 & s_{31} \\
s_2 & S & s_{12}
\end{Bmatrix} \\
&\quad \times (-)^{T_{12} + 2 t_3 + t_1 + t_2} \sqrt{(2 T_{12} + 1)(2 T_{31} + 1)} \begin{Bmatrix}
t_3 & t_1 & T_{31} \\
t_2 & T & T_{12}
\end{Bmatrix}
\end{align}
$$
## Faddeev Equation in Lagrange Mesh Basis

The Faddeev component can be expressed in the angular momentum basis and expanded using Lagrange mesh functions as:
$$
\langle f_{kx} f_{ky}\alpha_3 | \psi \rangle =  c_{k_x, k_y}^{\alpha_3} 
$$

For identical particles, the Faddeev equation can be rewritten as:
$$
\begin{align}
& \sum_{\alpha_3} \sum_{k_x, k_y}\left\langle f_{k_x'} f_{k_y'} \alpha_3' \middle| E - H_0 - V \middle| f_{k_x} f_{k_y} \alpha_3 \right\rangle \left\langle f_{k_x} f_{k_y} {\alpha_3}\middle|\psi \right\rangle  
= \sum_{\alpha_1} \sum_{i_x, i_y} \left\langle f_{k_x'}f_{k_y'} \alpha_3' \middle| V \middle|  f_{i_x} f_{i_y} {\alpha_1} \right\rangle \left\langle f_{i_x} f_{i_y} {\alpha_1}\middle|\psi \right\rangle \\
&\quad + \sum_{\alpha_2} \sum_{j_x, j_y}\left\langle f_{k_x'} f_{k_y'} {\alpha_3'} \middle| V \middle|   f_{j_x} f_{j_y} {\alpha_2} \right\rangle\left\langle f_{j_x} f_{j_y}{\alpha_2}\middle|\psi \right\rangle.
\end{align}
$$

This can be expressed more compactly as:
$$
\begin{align}
& \Bigg[ E [I_{\alpha_3}] \otimes [N_{k_x}] \otimes [N_{k_y}] - \sum_{\alpha_3}[\delta_{\alpha_3',\alpha_3}] \otimes [T_{k_x}^{\alpha_3}] \otimes [N_{k_y}] -\sum_{\alpha_3} [\delta_{\alpha_3',\alpha_3}] \otimes [N_{k_x}] \otimes [T_{k_y}^{\alpha_3}] \\&  - \sum_{\alpha_3',\alpha_3} [P_{\alpha_3',\alpha_3}]\otimes [V_{\alpha_3', \alpha_3}] \otimes [N_{k_y}^{\alpha_3}] \Bigg] c_{k_x, k_y}^{\alpha_3} 
= [V] \left( [R_{k_x, k_y, i_x, i_y}^{\alpha_3 \gets \alpha_1}] c_{i_x, i_y}^{\alpha_1} + [R_{k_x, k_y, j_x, j_y}^{\alpha_3 \gets \alpha_2}] c_{j_x, j_y}^{\alpha_2} \right),
\end{align}
$$
where $\otimes$ denotes the Kronecker product, $P_{\alpha_3',\alpha_3}[i,j]=\delta_{\alpha_3',i} \delta_{\alpha_3,j}$  

The Lagrange-Laguerre mesh regularized function is defined as:
$$
\begin{align}
f_{a_x}(x_{a'}) &= \lambda_{a_x}^{-1/2} \delta_{a, a'}, \quad a \in \{i, j, k\}, \\
f_{a_y}(y_{a'}) &= \lambda_{a_y}^{-1/2} \delta_{a, a'}, \quad a \in \{i, j, k\}.
\end{align}
$$

This function, denoted $f_j(x)$, is expressed as:
$$
f_j(x) = (-1)^j \left( h_N^\alpha x_j \right)^{-1/2} \frac{L_N^\alpha(x)}{x - x_j} x^{\alpha/2 + 1} e^{-x/2},
$$
where $L_N^\alpha(x)$ is the generalized Laguerre polynomial of order $N$ with parameter $\alpha$, $h_N^\alpha$ is the square norm of the polynomial, and $x_j$ are the roots of the polynomial. The generalized Laguerre polynomials $L_N^\alpha(x)$ are orthogonal polynomials that solve the Laguerre differential equation and are widely used in quantum mechanics, particularly for radial functions.

Since the regularized function is not orthogonal, the overlap matrices are:
$$
\begin{align}
[N_{a_x}] &= \langle f_{a_x'} | f_{a_x} \rangle = \delta_{a_x, a_x'} + \frac{(-)^{a_x - a_x'}}{\sqrt{x_{a_x} x_{a_x'}}}, \quad a \in \{i, j, k\}, \\
[N_{a_y}] &= \langle f_{a_y'} | f_{a_y} \rangle = \delta_{a_y, a_y'} + \frac{(-)^{a_y - a_y'}}{\sqrt{y_{a_y} y_{a_y'}}}, \quad a \in \{i, j, k\}.
\end{align}
$$

When the polynomial order $N$ is large, the roots $x_i$ may extend beyond the interaction range, leading to inefficient numerical integration. To improve efficiency, a scaling transformation is applied:
$$
r = h_S x,
$$
where $h_S$ is a scaling factor with units of femtometers (fm). The scaled basis is:
$$
\phi_i(r) = h_S^{-1/2} f_i(r/h_S),
$$
where the factor $h_S^{-1/2}$ ensures that the overlap $N_{ij}$ of the scaled basis matches that of the unscaled basis. Note that both the Lagrange-Laguerre function and the variable $x$ are dimensionless, while the scaling factor $h_S$ provides the physical unit.

For the kinetic energy operator, following Eq. (5.20) of Baye’s Physics Reports, we have:
$$
\begin{align}
[T_{a_x}] &= \frac{\hbar^2}{2 \mu_a^{2b} h_{S_x}^2} \left\{ \langle f_{a_x'} | -\frac{d^2}{dr^2} | f_{a_x} \rangle^G - (-)^{a_x - a_x'} \frac{1}{4 \sqrt{x_{a_x} x_{a_x'}}} + \frac{l_a (l_a + 1)}{x_{a_x}^2} \delta_{a_x, a_x'} \right\}, \\
[T_{a_y}] &= \frac{\hbar^2}{2 \mu_a^{3b} h_{S_y}^2} \left\{ \langle f_{a_y'} | -\frac{d^2}{dr^2} | f_{a_y} \rangle^G - (-)^{a_y - a_y'} \frac{1}{4 \sqrt{y_{a_y} y_{a_y'}}} + \frac{\lambda_a (\lambda_a + 1)}{y_{a_y}^2} \delta_{a_y, a_y'} \right\},
\end{align}
$$
with:
$$
\langle f_{a_x'} | -\frac{d^2}{dr^2} | f_{a_x} \rangle^G = \int_0^\infty dr \, f_{a_x'}(r) f_{a_x}''(r) =
\begin{cases}
-\frac{1}{12 x_{a_x}^2} \left[ x_{a_x}^2 - 2 (2N + \alpha + 1) x_{a_x} + \alpha^2 - 4 \right], & a_x = a_x', \\
(-1)^{a_x - a_x'} \frac{x_{a_x} + x_{a_x'}}{\sqrt{x_{a_x} x_{a_x'}} (x_{a_x} - x_{a_x'})^2}, & a_x \neq a_x',
\end{cases}
$$
$$
\langle f_{a_y'} | -\frac{d^2}{dr^2} | f_{a_y} \rangle^G = \int_0^\infty dr \, f_{a_y'}(r) f_{a_y}''(r) =
\begin{cases}
-\frac{1}{12 y_{a_y}^2} \left[ y_{a_y}^2 - 2 (2N + \alpha + 1) y_{a_y} + \alpha^2 - 4 \right], & a_y = a_y', \\
(-1)^{a_y - a_y'} \frac{y_{a_y} + y_{a_y'}}{\sqrt{y_{a_y} y_{a_y'}} (y_{a_y} - y_{a_y'})^2}, & a_y \neq a_y'.
\end{cases}
$$

For the potential matrix, two cases are considered. For a local potential $V_{\alpha_3', \alpha_3}$:
$$
[V_{\alpha_3', \alpha_3}] = \langle f_{a_x'} | V_{\alpha_3', \alpha_3} | f_{a_x} \rangle = \int_0^\infty f_{a_x'}(r) V_{\alpha_3', \alpha_3}(r) f_{a_x}(r) \, dr = V_{\alpha_3', \alpha_3}(x_{a_x}) \delta_{a_x, a_x'}.
$$

For a nonlocal potential:
$$
[V_{\alpha_3', \alpha_3}] = \langle f_{a_x'} | V_{\alpha_3', \alpha_3} | f_{a_x} \rangle = \int_0^\infty \int_0^\infty f_{a_x'}(r') V_{\alpha_3', \alpha_3}(r', r) f_{a_x}(r) \, dr' dr = \frac{V_{\alpha_3', \alpha_3}(x_{a_x'}, x_{a_x})}{f_{a_x'}(x_{a_x'}) f_{a_x}(x_{a_x})}.
$$

Finally, the transformation matrix is:
$$
\begin{align}
[R_{a_x, a_y, b_x, b_y}^{\alpha_a \gets \alpha_b}] &= \left\langle f_{a_x} f_{a_y} {\alpha_a} \middle|  f_{b_x} f_{b_y} {\alpha_b} \right\rangle \\
&= \int x_a^2 y_a^2 dx_a dy_a x_b^2 y_b^2dx_b dy_b \left\langle f_{a_x} f_{a_y} {\alpha_a} \middle| x_a y_a \alpha_a \right\rangle \left\langle x_a y_a \alpha_a \middle| x_b y_b \alpha_b \right\rangle \left\langle x_by_b\alpha_b \middle| f_{b_x} f_{b_y} {\alpha_b} \right\rangle \\
& = \int x_a^2 y_a^2 dx_a dy_a x_b^2 y_b^2dx_b dy_b \frac{f_{a_x}^{\alpha_a}f_{a_y}^{\alpha_a} }{x_a y_a} \left\langle x_a y_a \alpha_a \middle| x_b y_b \alpha_b \right\rangle \frac{f_{b_x}^{\alpha_b}f_{b_y}^{\alpha_b} }{x_b y_b}\\
&= \int d\cos\theta \, \frac{\mathcal{G}_{\alpha_a, \alpha_b}(x_a, y_a, \cos\theta) f_{b_x}^{\alpha_b}(\pi_b) f_{b_y}^{\alpha_b}(\xi_b) x_a y_a}{f_{a_x}^{\alpha_a}(x_a) f_{a_y}^{\alpha_a}(y_a) \pi_b \xi_b}.
\end{align}
$$
## Eigenvalue problem of the Faddeev Equations

After the Lagrange-mesh discretization the Faddeev equation reads
$$
E\,B\,[c] = \big(H_0 + V + V\cdot\mathcal{R} + V_{\mathrm{UIX}}\big)[c],
$$
where $B = I_{\alpha_3}\otimes N_{k_x}\otimes N_{k_y}$ is the overlap matrix, $H_0$ is the channel-diagonal kinetic part, $V$ is the full two-body potential matrix (with both within-channel and cross-channel blocks), $\mathcal{R}$ is the Jacobi-rearrangement matrix, and $V_{\mathrm{UIX}}$ is the optional three-body force.

Directly inverting $B$ (or the full operator $E B - H_0 - V$) at every energy is expensive, so one applies the **Malfiet-Tjon split**: keep the *full two-body potential* $V$ inside the operator to be inverted, and push only the *rearrangement term* and the *three-body force* into the right-hand side. This is allowed because $V$ has block structure that makes its inverse tractable: the two-body NN interaction conserves the **V-sector quantum numbers**
$$
q \;\equiv\; (J_{12},\, T_{12},\, s_{12},\, \lambda_3,\, J_3),
$$
so $V_{\alpha_3' \alpha_3}\neq 0$ only when $q(\alpha_3') = q(\alpha_3)$. (The remaining quantum numbers $J$, $\pi$, $T$, $M_T$, $s_1{=}s_2{=}s_3{=}1/2$, $t_1{=}t_2{=}t_3{=}1/2$ are global constants fixed once per calculation, so they do not need to be tracked in the per-channel sector key. The choice of the five per-channel labels above matches the V-coupling selection rules implemented in `swift/matrices.jl:235-246` and `swift/matrices_optimized.jl:540-555` exactly.)

Grouping the three-body channels into V-sectors $\mathcal{J}_q = \{\alpha_3 : q(\alpha_3) = q\}$, the matrix $V$ is **block-diagonal across V-sectors**. Within a sector $\mathcal{J}_q$, $V$ couples channels that share $(J_{12}, T_{12}, s_{12}, \lambda_3, J_3)$ but differ in the orbital quantum number $l_{12}$ — the canonical example is the ${}^3S_1\text{–}{}^3D_1$ tensor coupling in the deuteron sector ($J_{12}{=}1, T_{12}{=}0, s_{12}{=}1$), which mixes $l_{12}{=}0$ and $l_{12}{=}2$.

Define
$$
M(E) \;\equiv\; E\,B - H_0 - V, \qquad
\mathcal{S} \;\equiv\; V\cdot\mathcal{R} + V_{\mathrm{UIX}},
$$
so that
$$
[c] \;=\; M(E)^{-1}\,\mathcal{S}\,[c].
$$
With $V$ block-diagonal in the V-sector index $q$, $M(E)$ is itself block-diagonal in $q$ — the same structural property that made the strict channel-diagonal split work, just with **larger blocks** (size $n_q\cdot n_x\cdot n_y$ for a sector with $n_q$ coupled channels). Each $q$-block is inverted independently.

Guessing an eigenenergy $E_0$ reformulates the problem as the parametric eigenvalue equation
$$
\lambda(E_0)\,[c] \;=\; M(E_0)^{-1}\,\mathcal{S}\,[c]
\;\equiv\; K(E_0)\,[c].
$$
If $E_0$ is exactly the ground-state energy, then $\lambda(E_0)=1$; if it is below (above) the ground state, $\lambda(E_0)<1$ ($>1$). The split is **optimal in the sense that the RHS contains only the genuinely three-body content** — the rearrangement $\mathcal{R}$ that bridges Jacobi coordinate systems, and the irreducible three-nucleon force $V_{\mathrm{UIX}}$. All two-body physics, including tensor coupling, is captured analytically inside $M(E)^{-1}$.

> **Remark — special cases of the split.**
> Different choices of which subset of $V$ to absorb into $M$ trace out a family of preconditioners with the same algorithmic skeleton:
> - **Strict channel-diagonal** ($V \to V_{\alpha_3\alpha_3}$): $M$ is block-diagonal in the channel index $\alpha_3$ itself; RHS = $(V - V_{\alpha_3\alpha_3}) + V\cdot\mathcal{R} + V_{\mathrm{UIX}}$. Cheapest cache build (each block of $M$ is a single channel, $n_x\cdot n_y$ in size) but RHS still carries the tensor coupling.
> - **V-sector block-diagonal** ($V \to V$, the full V, exploiting that $V$ is block-diagonal in $q$): $M$ is block-diagonal in $q$ with blocks of size $n_q\cdot n_x\cdot n_y$; RHS = $V\cdot\mathcal{R} + V_{\mathrm{UIX}}$. Strictly better Faddeev kernel (RHS is smaller, $K(E)$ has smaller spectral radius → fewer Arnoldi steps to converge).
>
> **Current `swift/matrices.jl`** implements the **V-sector** special case (the 2026-06-15 cleanup deleted the strict channel-diagonal path); the strict channel-diagonal case is kept above only as the $n_q=1$ pedagogical limit. Both share the same Kronecker-product machinery developed in the next subsection — the only difference is the block size.

**Code reference:** the assembly of $\mathcal{S}$ is `precompute_RHS_cache` in `swift/MalflietTjon.jl` (V-sector variant: $\mathcal{S} = V\cdot\mathcal{R} + V_{\mathrm{UIX}}$, since the full $V$ is absorbed into $M$), and the analytical $M^{-1}$ is `M_inverse_operator_cached_vsector` (with `precompute_M_inverse_cache_vsector` / `MInverseCacheVSector` / `group_channels_by_v_sector`) in `swift/matrices.jl` (see the next subsection for the construction).

To find the ground state energy, one starts with an initial guess $E_0$ and uses the secant method to iteratively refine the energy. The secant method requires two initial points, $E_0$ and $E_1$, with their corresponding eigenvalues $\lambda(E_0)$ and $\lambda(E_1)$. The iterative formula for the secant method is:

$$
E_{n+1} = E_n - \frac{[\lambda(E_n) - 1](E_n - E_{n-1})}{\lambda(E_n) - \lambda(E_{n-1})}
$$

The iteration continues until $|\lambda(E_n) - 1| < \epsilon$, where $\epsilon$ is a predetermined tolerance. At convergence, $E_n$ approximates the ground state energy with $\lambda(E_n) \approx 1$.

---

### Arnoldi Method for the Faddeev Kernel

The numerical bottleneck in the above procedure is the evaluation of the dominant eigenvalue of the operator
$$
K(E) \;=\; M(E)^{-1}\,\mathcal{S} \;=\; \big[E B - H_0 - V_{\alpha_3\alpha_3}\big]^{-1}\,\Big[(V - V_{\alpha_3\alpha_3}) + V\cdot\mathcal{R} + V_{\mathrm{UIX}}\Big],
$$
which acts on the wavefunction $[c]$. Since $K(E)$ is a large but compact operator, direct diagonalization is unfeasible. Instead, one may employ the **Arnoldi method**, which is a Krylov-subspace projection technique. The action $K(E)\,v$ is **never assembled as a matrix** in the code — only the matrix-vector product is needed: apply $\mathcal{S}$ once (sparse-dense multiplication, since the assembled RHS matrix is $\sim 99\%$ zeros), then apply $M(E)^{-1}$ once via the four-step tensor-product algorithm of the previous subsection. The full per-Arnoldi-step cost is therefore $\mathcal{O}\big(n_\alpha\,n_x\,n_y\,(n_x+n_y)\big)$, with no $\mathcal{O}\big((n_\alpha n_x n_y)^3\big)$ inversion ever materialised.

The main idea is to generate an orthonormal basis $\{|\bar{\varphi}_i\rangle\}$ of the Krylov subspace
$$
\mathcal{K}_{\mathcal{N}} = \text{span}\{ |\bar\varphi_0\rangle, K|\bar\varphi_0\rangle, K^2|\bar\varphi_0\rangle, \ldots, K^{\mathcal{N}}|\bar\varphi_0\rangle \},
$$
starting from a normalized initial vector $|\bar\varphi_0\rangle$.

The algorithm proceeds as follows:

1. **Initialization**:  
   Choose a normalized starting vector $|\bar\varphi_0\rangle$, maybe in Gaussian form.

2. **Iteration**:  
   For $i = 0,1,\ldots,\mathcal{N}-1$, compute
   $$
   |\varphi_{i+1}\rangle = K|\bar\varphi_i\rangle,
   $$
   then orthogonalize against all previously constructed basis vectors and normalize:
   $$
   |\tilde\varphi_{i+1}\rangle = |\varphi_{i+1}\rangle - \sum_{j=0}^i |\bar\varphi_j\rangle \langle \bar\varphi_j | \varphi_{i+1}\rangle,
   $$
   $$
   |\bar\varphi_{i+1}\rangle = \frac{|\tilde\varphi_{i+1}\rangle}{\|\tilde\varphi_{i+1}\|}.
   $$

3. **Projection**:  
   Construct the Hessenberg matrix
   $$
   B_{ij} = \langle \bar\varphi_i | K | \bar\varphi_j \rangle,
   $$
   which is much smaller than the original kernel.

4. **Diagonalization**:  
   Solve the reduced eigenvalue problem
   $$
   B \, c = \lambda c.
   $$
   The resulting Ritz pairs $(\lambda, |\phi\rangle)$ approximate the true eigenpairs of $K$.

5. **Energy refinement**:  
   Select the Ritz value closest to unity, and use the associated Ritz vector to update the starting vector in the next secant iteration over $E$.





some notes on the comparison with Rimas' code, HAV is the scaling factor, N_REG is the regularized factor for $(\frac{x}{x_i})^{N_{REG}}$ 


## Normalization of the wave function

The wave function $|\Psi\rangle$ can be normalized using the condition:
$$
\langle \Psi | \Psi \rangle = \langle \Psi | (1 + P^+ + P^-) | \psi_3 \rangle = 3 \langle \Psi | \psi_3 \rangle,
$$
where $1 + P^+ + P^-$ represents the sum of the identity operator and the permutation operators $P^+$ and $P^-$, which exchange particles in a multi-particle system, and $|\psi_3\rangle$ is a reference state in the model space (e.g., a specific channel or basis state). The inner product $\langle \Psi | \Psi \rangle$ cannot be used directly for normalization because the permutation operators $P^+$ and $P^-$ may generate higher partial wave components that lie outside the truncated model space, leading to contributions that are not accounted for in the restricted basis. Instead, the inner product $\langle \Psi | \psi_3 \rangle$ provides a natural cutoff, as $|\psi_3\rangle$ is defined within the model space, ensuring that only relevant components are considered. The normalized wave functions are thus:
$$
\begin{align}
|\bar{\Psi}\rangle &= \frac{|\Psi\rangle}{\sqrt{3 \langle \Psi | \psi_3 \rangle}}, \\
|\bar{\psi}_3\rangle &= \frac{|\psi_3\rangle}{\sqrt{3 \langle \Psi | \psi_3 \rangle}}.
\end{align}
$$

### Computation of Expectation Values

To compute the expectation value of the kinetic energy for the normalized wave function $|\bar{\Psi}\rangle$, we have:
$$
\langle \bar{\Psi} | T | \bar{\Psi} \rangle = \langle \bar{\Psi} | T (1 + P^+ + P^-) | \bar{\psi}_3 \rangle = 3 \langle \bar{\Psi} | T | \bar{\psi}_3 \rangle,
$$
where $T$ is the kinetic energy operator, and the equality follows from the action of the operator $1 + P^+ + P^-$ on the reference state $|\bar{\psi}_3\rangle$, consistent with the structure of the wave function in the model space.

For the potential energy, the expectation value is:
$$
\langle \bar{\Psi} | V | \bar{\Psi} \rangle = 3 \langle \bar{\Psi} | V_{12} | \bar{\Psi} \rangle,
$$
where $V$ is the potential energy operator, and $V_{12}$ represents the two-body interaction between particles 1 and 2 (or a specific pair in the system). This simplification holds because the potential $V$ is typically defined within the model space, and the channel basis introduces a natural cutoff that restricts the interaction to the relevant components of the wave function.


## Lagrange function definition

Based on the definition of Daniel Baye, the regularized Lagrange Laguerre function is defined as 
$$
f_i(x)=(-1)^i\left(h_N^\alpha x_i\right)^{-1 / 2} \frac{L_N^\alpha(x)}{x-x_i} x^{\alpha / 2+1} e^{-x / 2} .
$$
This definition guarantee the $f_i(x_i)=(\lambda_i)^{-1/2}$, this value to be positive. On the other hand,  Riams use relation of 

## Partial wave decomposition of UIX force 
The partial-wave decomposition of the three-body force into the Jacobi partial-wave basis differs from that of two-body forces. As an example, let us consider the Urbana IX (UIX) force, following the formulation given in the PhD thesis of Rimas. The complete UIX force between particles $(123)$ can be written as
$$
\begin{gathered} 
V_{123}=A_{2 \pi}\left(X_{31} X_{12} I_{23}^{-}+X_{12} X_{31} I_{23}^{+}+X_{12} X_{23} I_{31}^{-}+X_{23} X_{12} I_{31}^{+}+X_{23} X_{31} I_{12}^{-}+X_{31} X_{23} I_{12}^{+}\right) \\ 
+U_{0} T^{2}\left(r_{31}\right) T^{2}\left(r_{12}\right)+U_{0} T^{2}\left(r_{12}\right) T^{2}\left(r_{23}\right)+U_{0} T^{2}\left(r_{23}\right) T^{2}\left(r_{31}\right) 
\end{gathered}  
$$
The decomposed two-body-like part can be defined as
$$
V_{12}^{(3)}=A_{2 \pi}\left(X_{12} X_{31} I_{23}^{+}+X_{12} X_{23} I_{31}^{-}\right)+\tfrac{1}{2} U_{0}\left[T^{2}\left(r_{12}\right) T^{2}\left(r_{23}\right)+T^{2}\left(r_{31}\right) T^{2}\left(r_{12}\right)\right] .
$$
with
$$
\begin{aligned} 
I_{23}^{-}&=2\left(\vec{\tau}_{2} \cdot \vec{\tau}_{3}-\tfrac{i}{4} \vec{\tau}_{1} \cdot \vec{\tau}_{2} \times \vec{\tau}_{3}\right), \\ 
I_{31}^{-}&=2\left(\vec{\tau}_{3} \cdot \vec{\tau}_{1}-\tfrac{i}{4} \vec{\tau}_{2} \cdot \vec{\tau}_{3} \times \vec{\tau}_{1}\right), \\ 
I_{23}^{+}&=2\left(\vec{\tau}_{2} \cdot \vec{\tau}_{3}+\tfrac{i}{4} \vec{\tau}_{1} \cdot \vec{\tau}_{2} \times \vec{\tau}_{3}\right) . 
\end{aligned}  
$$
The force is explicitly defined as a superposition of NN-like interactions, derived from $\pi$-exchange NN forces:
$$
X_{i j}=Y\left(r_{i j}\right) \vec{\sigma}_{i} \cdot \vec{\sigma}_{j}+T\left(r_{i j}\right) S_{i j} ,
$$
which contains both a spin–spin part $\vec{\sigma}*{i} \cdot \vec{\sigma}*{j}$ and a tensor part,
$$
S_{i j}=3\left(\vec{\sigma}_{i} \cdot \vec{r}_{i j}\right)\left(\vec{\sigma}_{j} \cdot \vec{r}_{i j}\right)-\vec{\sigma}_{i} \cdot \vec{\sigma}_{j}.
$$
The radial functions are given by
$$
\begin{aligned} 
Y(r) & =\frac{e^{-m_{\pi} r}}{m_{\pi} r}\left(1-e^{-c r^{2}}\right), \\ 
T(r) & =\left[1+\frac{3}{m_{\pi} r}+\frac{3}{(m_{\pi} r)^{2}}\right] \frac{e^{-m_{\pi} r}}{m_{\pi} r}\left(1-e^{-c r^{2}}\right)^{2}. 
\end{aligned}  
$$
The Urbana IX model is characterized by three fitted parameters with the following values:
$A_{2 \pi}=-0.0293~\mathrm{MeV}$, $U_{0}=0.0048~\mathrm{MeV}$, and $c=2.1~\mathrm{fm}^{-2}$. $m_\pi=\frac{1}{3}m_{\pi^0}+\frac{2}{3}m_{\pi^{\pm}}$ 
In the Faddeev equation, one has to compute the matrix element
$$
\begin{aligned}
\langle x_3 y_3 \alpha_3 | V_{12}^{(3)}(1+P^+ +P^-) | \psi_3 \rangle 
&= \langle x_3 y_3 \alpha_3 | V_{12}^{(3)} | \Psi\rangle \\
&= A_{2 \pi}\left\langle x_3 y_3 \alpha_3\right| X_{12} X_{31} I_{23}^{+}+X_{12} X_{23} I_{31}^{-}|\Psi\rangle \\
&\quad +\tfrac{1}{2} U_{0}\left\langle x_3 y_3 \alpha_3\right| T^{2}\left(r_{12}\right) T^{2}\left(r_{23}\right)+T^{2}\left(r_{31}\right) T^{2}\left(r_{12}\right)|\Psi\rangle .
\end{aligned}
$$
We now focus on the first two terms; the purely radial terms will be discussed separately. In the current implementation, the potential is evaluated in the Lagrange basis, giving
$$
\begin{aligned}
& \left\langle f_{k_x}f_{k_y} \alpha_3 \middle|X_{12}X_{31}I_{23}^+ + X_{12}X_{23}I_{31}^- \middle|\Psi \right\rangle \\
&= \sum_{k_x'k_y'\alpha_3'}\langle f_{k_x} f_{k_y} \alpha_3 | X_{12} | f_{k_x'} f_{k_y'} \alpha_3'\rangle   \\
&\quad \times \Biggl[\sum_{j_xj_y\alpha_2}\langle f_{k_x'} f_{k_y'}\alpha_3' | I_{23}^+ | f_{j_x}f_{j_y}\alpha_2\rangle \langle f_{j_x}f_{j_y}\alpha_2|X_{31}|\Psi\rangle \\
&\qquad + \sum_{i_x i_y\alpha_1} \langle f_{k'_x}f_{k'_y}\alpha_3'| I_{31}^-|f_{i_x}f_{i_y}\alpha_1\rangle \langle f_{i_x}f_{i_y}\alpha_1 |X_{23}  | \Psi\rangle \Biggr] .
\end{aligned}
$$
Here $I$ acts only in isospin space, while $X$ contains spin operators; thus $I$ commutes with $X$. The intermediate states are introduced to allow evaluation of $X$ and $T$ in their natural Jacobi coordinates. Importantly, the intermediate basis must be complete: because the potential is symmetric under exchange of pair particles, even unphysical (non-antisymmetric) states must be included.
It can be shown that
$$
\langle f_{k_x'} f_{k_y'}\alpha_3' | I_{23}^+ | f_{j_x}f_{j_y}\alpha_2\rangle
=\epsilon_{\alpha_3'}\epsilon_{\alpha_1} \langle f_{k'_x}f_{k'_y}\alpha_3'| I_{31}^-|f_{i_x}f_{i_y}\alpha_1\rangle ,
$$
so that
$$
\begin{aligned}
& \left\langle f_{k_x}f_{k_y} \alpha_3 \middle|X_{12}X_{31}I_{23}^+ + X_{12}X_{23}I_{31}^- \middle|\Psi \right\rangle \\
&= (1+\epsilon_{\alpha_3'}\epsilon_{\alpha_1})
\sum_{k_x'k_y'\alpha_3'}\langle f_{k_x} f_{k_y} \alpha_3 | X_{12} | f_{k_x'} f_{k_y'} \alpha_3'\rangle   
\sum_{i_x i_y\alpha_1} \langle f_{k'_x}f_{k'_y}\alpha_3'| I_{31}^-|f_{i_x}f_{i_y}\alpha_1\rangle \langle f_{i_x}f_{i_y}\alpha_1 |X_{23}  | \Psi\rangle .
\end{aligned}
$$
Here $\epsilon_{\alpha}$ is the symmetry factor of the state $\alpha$, defined as $\epsilon=(-1)^{\ell+s+t}$.
It is worth noting that multiplication with $I_{23}^{+}$ or $I_{31}^{-}$ destroys the antisymmetry of the Faddeev components. However, since the operators $X_{ij}$ are symmetric under exchange of $(ij)$ and the full wave function $\Psi$ is antisymmetric, the states $\alpha_1$ remain antisymmetric. Similarly, $\alpha_3'$ is antisymmetric under $(12)$, because the operator $X_{12} X_{31} I_{23}^{+}+X_{12} X_{23} I_{31}^{-}$ is symmetric under $(12)$. As a result, contributions from non-physical (symmetric) states cancel, while contributions from physical states add coherently. Therefore, one may restrict the sum to physical states only and replace the prefactor $\left[1+\epsilon_{\alpha_3'}\epsilon_{\alpha_1}\right]$ by $2$.
The other two potential terms can be treated in the same way:
$$
\begin{aligned}
&\left\langle f_{k_x}f_{k_y} \alpha_3 \middle| T^{2}\left(r_{12}\right) T^{2}\left(r_{23}\right)+T^{2}\left(r_{31}\right) T^{2}\left(r_{12}\right)\middle|\Psi \right\rangle \\
&= (1+\epsilon_{\alpha_3'}\epsilon_{\alpha_1}) \sum_{k_x'k_y'\alpha_3'}\sum_{i_xi_y\alpha_1}
\langle f_{k_x}f_{k_y} \alpha_3 | T^2(r_{12})|f_{k'_x}f_{k_y'}\alpha_3'\rangle \\
&\quad \times  \langle f_{k'_x}f_{k_y'}\alpha_3'| f_{i_x}f_{i_y}\alpha_1\rangle \langle f_{i_x}f_{i_y}\alpha_1 | T^2(r_{23})|\Psi\rangle .
\end{aligned}
$$
The same reasoning applies: one can restrict to physical intermediate states, accounting for the doubled contribution.
Next, we consider the NN-like operator $X_{ij}$. Its matrix elements are
$$
\begin{aligned}
\langle xy\alpha_3 | X_{12}|x'y'\alpha'_3\rangle 
&= \frac{\delta(x-x')}{xx'} \frac{\delta(y-y')}{yy'} 
\delta_{s_{12},s_{12}'} \delta_{J_{12},J_{12}'} \delta_{\lambda_3,\lambda_3'}\delta_{J_3,J_3'}\delta_{J,J'} \delta_{T_{12},T_{12}'}\delta_{T,T'} \\
&\quad \times \Bigl[\delta_{l_{12},l_{12}'}(4s_{12}-3)Y(x)+\delta_{s_{12},1}T(x)S_{l_{12},l_{12}',J_{12}}\Bigr] .
\end{aligned}
$$
where
$$
S_{l_{12},\, l'_{12},\, J_{12}} =
\left[
\begin{array}{c|ccc}
 & l'_{12} = J_{12} - 1 & l'_{12} = J_{12} & l'_{12} = J_{12} + 1 \\ \hline
l_{12} = J_{12} - 1 &
\dfrac{-2\,(J_{12}-1)}{2J_{12}+1} & 0 & \dfrac{6\sqrt{J_{12}(J_{12}+1)}}{2J_{12}+1} \\
l_{12} = J_{12} &
0 & 2 & 0 \\
l_{12} = J_{12} + 1 &
\dfrac{6\sqrt{J_{12}(J_{12}+1)}}{2J_{12}+1} & 0 & \dfrac{-2\,(J_{12}+2)}{2J_{12}+1}
\end{array}
\right] .
$$
And it is easy to prove that 
$$
\begin{aligned}
\langle f_{k_x} f_{k_y} \alpha_3 | X_{12} | f_{k_x'} f_{k_y'} \alpha_3' \rangle &=\delta_{k_x,k_x'}\delta_{k_y,k_y'}
\delta_{s_{12},s_{12}'} \delta_{J_{12},J_{12}'} \delta_{\lambda_3,\lambda_3'}\delta_{J_3,J_3'}\delta_{J,J'} \delta_{T_{12},T_{12}'}\delta_{T,T'} \\
&\quad \times \Bigl[\delta_{l_{12},l_{12}'}(4s_{12}-3)Y(x_{k_x})+\delta_{s_{12},1}T(x_{k_x})S_{l_{12},l_{12}',J_{12}}\Bigr] .
\end{aligned}
$$


The isospin operators evaluate as
$$
\left\langle\left(\left(t_{1} t_{2}\right)_{T_{12}} t_{3}\right)_{T}\right| 
\vec{\tau}_{3} \cdot \vec{\tau}_{1}
\left|\left(\left(t_{2} t_{3}\right)_{T_{23}} t_{1}\right)_{T^{\prime}}\right\rangle  
=\delta_{T T^{\prime}}(-)^{T_{12}+1} 
6 \sqrt{\hat{T}_{12} \hat{T}_{23}}
\left\{
\begin{array}{ccc}
\tfrac{1}{2} & \tfrac{1}{2} & T_{23} \\
\tfrac{1}{2} & 1 & \tfrac{1}{2} \\
T_{12} & \tfrac{1}{2} & T
\end{array}
\right\},
$$
and
$$
\begin{aligned}
&-\tfrac{i}{4}\left\langle\left(\left(t_{1} t_{2}\right)_{T_{12}} t_{3}\right)_{T}\right| 
\vec{\tau}_{2} \cdot \vec{\tau}_{3} \times \vec{\tau}_{1}
\left|\left(\left(t_{2} t_{3}\right)_{T_{23}} t_{1}\right)_{T^{\prime}}\right\rangle \\
&=\delta_{T T^{\prime}} 
6 \sqrt{\hat{T}_{12} \hat{T}_{23}} 
\sum_{\xi }(-)^{2 T-\xi+\tfrac{1}{2}}
\left\{
\begin{array}{ccc}
\xi & \tfrac{1}{2} & 1 \\
\tfrac{1}{2} & \tfrac{1}{2} & T_{12}
\end{array}
\right\}
\left\{
\begin{array}{ccc}
T & \tfrac{1}{2} & T_{12} \\
\tfrac{1}{2} & 1 & \xi \\
T_{23} & \tfrac{1}{2} & \tfrac{1}{2}
\end{array}
\right\}.
\end{aligned}
$$



## Tensor-product construction of $M^{-1}$

We work in the V-sector decomposition (this is what `swift/matrices.jl` implements; the algorithm reduces to the strict channel-diagonal case when each sector contains a single channel, and the algebra below is written for general sector size and reduces correctly to $n_q = 1$).

Group the three-body channels by their V-sector index $q$ defined above, $\mathcal{J}_q = \{\alpha_3 : q(\alpha_3) = q\}$, with $n_q = |\mathcal{J}_q|$. Then $M(E) = E\,B - H_0 - V$ is **block-diagonal across the sectors $q$**:
$$
M(E) \;=\; \bigoplus_q M^{(q)}(E),
\qquad
M^{(q)}(E) \in \mathbb{C}^{(n_q n_x n_y)\times(n_q n_x n_y)}.
$$

A useful technical assumption (satisfied for the V-sector definition adopted here) is that within a single sector $q$ all channels share the same orbital quantum number $\lambda_3$, so the $y$-kinetic matrix $T_{k_y}$ is sector-uniform; $V$ depends only on the pair coordinate $x_3$, so it couples the channels of $\mathcal{J}_q$ through an $(n_q\cdot n_x)\times(n_q\cdot n_x)$ matrix $V^{(q)}$ that is *diagonal in the $y$-mesh*. Each sector block then takes the **Kronecker-separable** form
$$
M^{(q)}(E) \;=\; E\,\big(I_{n_q}\otimes N_{k_x}\otimes N_{k_y}\big) \;-\; \mathcal{H}^{(q)}_{x}\otimes N_{k_y} \;-\; \big(I_{n_q}\otimes N_{k_x}\big)\otimes T_{k_y},
$$
with the **coupled** $x$-Hamiltonian-like matrix
$$
\mathcal{H}^{(q)}_{x}
\;=\;
\underbrace{\bigoplus_{a\in\mathcal{J}_q} T_{k_x}^{a}}_{\text{block-diag in channels}}
\;+\;
\underbrace{V^{(q)}}_{\text{couples channels}}
\;\in\;\mathbb{C}^{(n_q n_x)\times(n_q n_x)}.
$$
This is the key generalization over the strict channel-diagonal split: $V^{(q)}$ now includes **all** $V_{a',a}$ matrix elements between channels $a,a' \in \mathcal{J}_q$ (in particular the ${}^3S_1\text{–}{}^3D_1$ tensor coupling for the $J_{12}=1, T_{12}=0$ sector), instead of restricting to $a' = a$.

Pulling $\big(I_{n_q}\otimes N_{k_x}\otimes N_{k_y}\big)$ out on the left,
$$
M^{(q)}(E) \;=\; \big(I_{n_q}\otimes N_{k_x}\otimes N_{k_y}\big)\,\Big[\,E\,I \;-\; \big(I_{n_q}\otimes N_{k_x}\big)^{-1}\mathcal{H}^{(q)}_{x}\otimes I_{k_y} \;-\; I^{(n_q)}_{x}\otimes \big(N_{k_y}^{-1} T_{k_y}\big)\,\Big],
$$
where $I^{(n_q)}_x \equiv I_{n_q}\otimes I_{k_x}$ and $\big(I_{n_q}\otimes N_{k_x}\big)^{-1} = I_{n_q}\otimes N_{k_x}^{-1}$.

Diagonalize the two reduced matrices:
$$
\big(I_{n_q}\otimes N_{k_x}^{-1}\big)\,\mathcal{H}^{(q)}_{x} \;=\; \mathcal{U}^{(q)}_{x}\,d^{(q)}_{x}\,\big(\mathcal{U}^{(q)}_{x}\big)^{-1},
\qquad
N_{k_y}^{-1}\, T_{k_y} \;=\; U_{y}\,d_{y}\,U_{y}^{-1},
$$
with $d^{(q)}_{x}$ (size $n_q n_x$) and $d_{y}$ (size $n_y$) diagonal. The columns of $\mathcal{U}^{(q)}_{x}$ are joint $(N_{k_x},$ channel-mixed$)$-orthogonal; the columns of $U_{y}$ are $N_{k_y}$-orthogonal. The block then factorises as
$$
M^{(q)}(E) \;=\; \underbrace{\big(I_{n_q}\otimes N_{k_x}\otimes N_{k_y}\big)}_{N^{(q)}}\;\underbrace{\big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)}_{U^{(q)}}\;D^{(q)}(E)\;\underbrace{\big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)^{-1}}_{(U^{(q)})^{-1}},
$$
with the **purely-diagonal** central factor (of dimension $n_q n_x n_y$)
$$
D^{(q)}(E)\;=\; E\,I - d^{(q)}_{x}\otimes I_{y} - I^{(q)}_{x}\otimes d_{y},
\qquad
\big[D^{(q)}(E)\big]_{(\mu,i_y),(\mu,i_y)} = E - d^{(q)}_{x}[\mu] - d_{y}[i_y],
$$
where $\mu = 1,\ldots,n_q n_x$ enumerates the joint (channel, $k_x$) eigenmode of $\mathcal{H}^{(q)}_x$. Inverting block-by-block (the sector blocks decouple by construction) gives
$$
\boxed{\;\;M(E)^{-1}\big|_{q} \;=\; U^{(q)}\;D^{(q)}(E)^{-1}\;(U^{(q)})^{-1}\;(N^{(q)})^{-1} \;=\; \big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)\;D^{(q)}(E)^{-1}\;\big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)^{-1}\;\big(I_{n_q}\otimes N_{k_x}^{-1}\otimes N_{k_y}^{-1}\big).\;\;}
$$

> **Reduction to the strict channel-diagonal case.** Setting $n_q = 1$ (each sector contains a single channel $\alpha_3$) collapses $\mathcal{H}^{(q)}_x \to T_{k_x}^{\alpha_3} + V_{\alpha_3\alpha_3}$ and $\mathcal{U}^{(q)}_x \to U_{\alpha_3}^x$, recovering the old strict channel-diagonal formula. The implemented code keeps the general $n_q \ge 1$ V-sector form, so sectors with tensor coupling (e.g. the $J_{12}=1, T_{12}=0$ deuteron sector) are handled exactly.

### Application to a vector (algorithm)

The action $w = M(E)^{-1}\,v$ is implemented sector-by-sector as the four sequential operations (right-to-left in the boxed expression above):

For each V-sector $q$:

1. Extract the sector block $v^{(q)} \in \mathbb{C}^{n_q n_x n_y}$ by gathering the entries of $v$ belonging to channels in $\mathcal{J}_q$.
2. $t_1 \;=\; \big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)^{-1}\,\big(I_{n_q}\otimes N_{k_x}^{-1}\otimes N_{k_y}^{-1}\big)\,v^{(q)}$.
3. $t_2 \;=\; D^{(q)}(E)^{-1}\odot t_1$ — element-wise multiplication by the precomputed sector diagonal.
4. $w^{(q)} \;=\; \big(\mathcal{U}^{(q)}_{x}\otimes U_{y}\big)\,t_2$.
5. Scatter $w^{(q)}$ back into the global vector $w$ at the channel positions of $\mathcal{J}_q$.

In the current `swift/matrices.jl` ($n_q=1$ for every sector) this collapses to the per-channel loop on lines `684-693`, with `U_blocks[α]` $=U_{\alpha_3}^x\otimes U_y^{\alpha_3}$, `U_inv_N_inv_blocks[α]` $= (U_{\alpha_3}^x\otimes U_y^{\alpha_3})^{-1}(N_{k_x}^{-1}\otimes N_{k_y}^{-1})$, and `D_inv_blocks[α]` the diagonal $D^{(q)}(E)^{-1}$. The V-sector variant replaces the per-channel arrays by per-sector arrays of larger size $n_q n_x \times n_q n_x$ (in $x$) and the same $n_y\times n_y$ (in $y$).

### Caching strategy

The decomposition cleanly separates energy-independent and energy-dependent work, which is exploited by `precompute_M_inverse_cache_vsector` / `M_inverse_operator_cached_vsector` in `swift/matrices.jl`. The table is written for the V-sector variant, with $\alpha_3 \to q$ and per-channel size $n_x\times n_x \to$ per-sector size $n_q n_x \times n_q n_x$ in the $x$-direction (the $n_q=1$ limit recovers the old channel-diagonal form):

| Quantity | Depends on $E$? | Stored in (current code, $n_q{=}1$) | Recomputed when |
|---|---|---|---|
| $N_{k_x}^{-1}, N_{k_y}^{-1}$ | no | `Nx_inv`, `Ny_inv` (single $n_x\times n_x$ + $n_y\times n_y$) | basis change |
| $\mathcal{U}^{(q)}_x, U_y, d^{(q)}_x, d_y$ (per sector) | no | `Ux_arrays`, `Uy_arrays`, `dx_arrays`, `dy_arrays` | basis or potential change |
| $\mathcal{U}^{(q)}_x \otimes U_y$ | no | `U_blocks[α]` (current) → `U_blocks[q]` (V-sector) | basis change |
| $\big(\mathcal{U}^{(q)}_x \otimes U_y\big)^{-1}\,\big(I_{n_q}\otimes N_{k_x}^{-1}\otimes N_{k_y}^{-1}\big)$ | no | `U_inv_N_inv_blocks[α]` (current) → `[q]` (V-sector) | basis change |
| $D^{(q)}(E)^{-1}$ (diagonal, $n_q n_x n_y$ entries per sector) | **yes** | `D_inv_blocks[α]` (current) → `[q]` (V-sector) | every secant step in $E$ |

The energy update step is therefore $\mathcal{O}\big(\sum_q n_q n_x n_y\big) = \mathcal{O}(n_\alpha n_x n_y)$ — basis-size in total, essentially free compared with the one-time $\mathcal{O}\big(\sum_q (n_q n_x)^3 + n_y^3\big)$ cache build. For the $\mathrm{jx\,max}=6$ tritium calculation ($n_\alpha=50$, $n_x=n_y=20$) the current channel-diagonal cache build is `Precomputing M⁻¹ cache (one-time)... 0.06 s`; the V-sector upgrade would slightly increase this (the largest sectors have $n_q=2$, so eigendecomposition cost $\sim 8\times$ per coupled sector), still well under a second total.

### Generalisation to complex scaling

Under the rotation $r \to r\,e^{i\theta}$ the kinetic matrices and potential become complex; $T_x$, $T_y$, $V$ and therefore $N^{-1}\mathcal{H}^{(q)}_x$, $N^{-1}T_y$ all live in $\mathbb{C}^{n\times n}$, but their eigendecomposition is structurally identical. The cache type is generic: `MInverseCacheVSector{T}` with `T ∈ {Float64, ComplexF64}` (`matrices.jl`); the same algorithm runs for $\theta=0$ (real) and $\theta\neq 0$ (complex) without code changes.



# Scattering with Complex scaling
The Faddeev component can be split into two terms as $\psi_3=\psi_3^{in}+\psi_3^{sc}$, containing the plane wave $\psi_3^{in}(\vec{x}_3,\vec{y}_3,\vec{k}) =\phi_{12}(\vec{x}_3) e^{i\vec{k}\cdot\vec{y}_3}$, where $\phi_{12}$ is the bound state of the projectile and $\psi_3^{sc}$ is the scattering wave function which can be rewritten as 
$$
\psi_{3}^{\mathrm{sc}}\left(\mathbf{x}_3, \mathbf{y}_3\right) \underset{y_3 \rightarrow \infty}{=}  A\left(\widehat{x}_3, \widehat{y}_3, x_3 / y_3\right) \frac{\exp (i K \rho)}{\rho^{5 / 2}} 
+\sum_n f_{n}\left(\widehat{y}_3\right) \phi_n\left(\mathbf{x}_3\right) \frac{\exp \left(i q_n y_3\right)}{\left|y_3\right|}
$$
where $\rho=\sqrt{x_3^2 + y_3^2}$ is the hyperradius, $A\left(\widehat{x}_3, \widehat{y}_3, x_3 / y_3\right)$ is the three-particle breakup amplitude, and $f_{n}\left(\widehat{y}_3\right)$ denotes the two-body transition amplitude from the initial channel to channel $n$. In this expression, the sum runs over all open binary channels $n$. $K$ is the three-particle breakup momentum satisfying the energy conservation relation $E=\frac{\hbar^2}{m}K^2$, where $E$ is the three-body total energy and $m$ is the mass of each particle. Then, for the three-body problem, separating the incoming wave of the particles scattered on a bound pair results in a single equation, 
$$
 {\left[E-H_0-V_{12}\left(\mathbf{x}_3\right)\right] \psi_{3}^{\mathrm{sc}}\left(\mathbf{x}_3, \mathbf{y}_3\right)-V_{12}\left(\mathbf{x}_3\right) \sum_{j \neq 3} \psi_{j}^{\mathrm{sc}}\left(\mathbf{x}_j, \mathbf{y}_j\right)} 
 =V_{12}\left(\mathbf{x}_3\right) \sum_{j \neq 3} \psi_j^{in}(\mathbf{x}_j, \mathbf{y}_j)
$$
and 
$$
[E-H_0-V_{12}\left(\mathbf{x}_3\right)] \psi_3^{in} =0 
$$
It should also be noted that the equal mass case is different compared with the unequal mass case. For unequal masses, the system has a unique incoming configuration, which leads to one plane-wave source term. However, for equal masses, all particles are equivalent, resulting in three indistinguishable incoming channels, and therefore the two extra terms must be added to preserve symmetry. **And note that for $n+d$ scattering or $p+d$ scattering there is no bound state of $2n$ or $2p$, so there is only one inhomogeneous term.** 
One may easily see that the scattered part of the Faddeev amplitude vanishes for large hyperradius if the particle coordinates are properly complex scaled, $\bar{\mathbf{x}}_3=\mathbf{x}_3e^{i\theta}$, $\bar{\mathbf{y}}_3=\mathbf{y}_3e^{i\theta}$, and $\bar{\rho}=\rho e^{i\theta}$. However, in order to obtain a solution of the problem on a finite grid, one should ensure the inhomogeneous term on the right-hand side of the Faddeev equation also vanishes outside the resolution domain. This is easy to verify: since the interaction term $V_{12}$ is short-ranged, the RHS term is damped by it.

## Scattering Equation in Lagrange Mesh Basis

The Faddeev scattering equation can be expressed in the angular momentum basis using Lagrange mesh functions as
$$
\begin{align}
& \sum_{\alpha_3} \sum_{k_x k_y} \bigl\langle f_{k_x'} f_{k_y'} \alpha_3' \bigm| E - H_0 - V \bigm| f_{k_x} f_{k_y} \alpha_3 \bigr\rangle \bigl\langle f_{k_x} f_{k_y} \alpha_3 \bigm| \psi^{\text{sc}} \bigr\rangle \\
&{}- \sum_{\alpha_1} \sum_{i_x i_y} \bigl\langle f_{k_x'} f_{k_y'} \alpha_3' \bigm| V \bigm| f_{i_x} f_{i_y} \alpha_1 \bigr\rangle \bigl\langle f_{i_x} f_{i_y} \alpha_1 \bigm| \psi^{\text{sc}} \bigr\rangle \\
&{}- \sum_{\alpha_2} \sum_{j_x j_y} \bigl\langle f_{k_x'} f_{k_y'} \alpha_3' \bigm| V \bigm| f_{j_x} f_{j_y} \alpha_2 \bigr\rangle \bigl\langle f_{j_x} f_{j_y} \alpha_2 \bigm| \psi^{\text{sc}} \bigr\rangle \\
&= \sum_{\alpha_1} \sum_{i_x i_y} \bigl\langle f_{k_x'} f_{k_y'} \alpha_3' \bigm| V \bigm| f_{i_x} f_{i_y} \alpha_1 \bigr\rangle \bigl\langle f_{i_x} f_{i_y} \alpha_1 \bigm| \psi^{\text{in}} \bigr\rangle. \\
&{}+ \sum_{\alpha_2} \sum_{j_x j_y} \bigl\langle f_{k_x'} f_{k_y'} \alpha_3' \bigm| V \bigm| f_{j_x} f_{j_y} \alpha_2 \bigr\rangle \bigl\langle f_{j_x} f_{j_y} \alpha_2 \bigm| \psi^{\text{in}} \bigr\rangle
\end{align}
$$

This equation can be written in compact matrix form as
$$
\begin{align}
\Bigg[ E \,[I_{\alpha_3}] &\otimes [N_{k_x}] \otimes [N_{k_y}] 
{}- \sum_{\alpha_3} [\delta_{\alpha_3',\alpha_3}] \otimes [T_{k_x}^{\alpha_3}] \otimes [N_{k_y}] 
{}- \sum_{\alpha_3} [\delta_{\alpha_3',\alpha_3}] \otimes [N_{k_x}] \otimes [T_{k_y}^{\alpha_3}] \\
&{}- [V] \Bigl( [I] + [R_{k_x, k_y, i_x, i_y}^{\alpha_3 \gets \alpha_1}] + [R_{k_x, k_y, j_x, j_y}^{\alpha_3 \gets \alpha_2}] \Bigr) \Bigg] [c] = [b],
\end{align}
$$
where the only difference from the bound-state problem is the inhomogeneous term
$$
[b] =2* [V] [R_{k_x, k_y, i_x, i_y}^{\alpha_3 \gets \alpha_1}] [\varphi].
$$
For the bound-state problem, the RHS is zero. The source term components are
$$
\varphi_i = \bigl\langle f_{i_x} f_{i_y} \alpha_1 \bigm| \psi^{\text{in}} \bigr\rangle 
= \int \frac{f_{i_x}(x_i) f_{i_y}(y_i)}{x_i y_i} \cdot \frac{\phi_d^{\alpha_1}(x_i) F^{\alpha_1}(y_i)}{x_i y_i} \, x_i^2 y_i^2 \, dx_i \, dy_i = \frac{\phi_d^{\alpha_1}(x_i) F^{\alpha_1}(y_i)}{f_{i_x}(x_i) f_{i_y}(y_i)},
$$
where $F^{\alpha_1}(y_i)$ is the regular part of the Coulomb wave function in the partial-wave basis.

## Apply the Complex Scaling Method 

The complex scaling operation transform the coordinate from real axis into complex by 
$$
r \to r e^{i\theta}
$$
where $r$ is the radial coordinate and $\theta$ is a scaling angle. This transformation rotates the coordinate $r$ by an angle $\theta$ in the complex plane. The corresponding operator for this rotation transformation can be expressed as 
$$
\hat{S}(\theta) = e^{i\theta/2} e^{i \theta x \frac{\partial}{\partial x}}
$$
where $\hat{S}(\theta)$ is the scaling operator. When we apply this operator to a wave function $\Psi(r)$, we obtain
$$
\hat{S}(\theta) \Psi(r) = e^{i\theta/2} \Psi(re^{i\theta})
$$
##  Scaling the Potential

When an analytical expression for the potential is not available—such as when the potential is obtained through a folding procedure—direct rotation of the potential function becomes challenging. To address this, a backward rotation can be applied to the basis functions.

Assuming that only discrete values of the potential are available at specific mesh points along the real axis, the integral can be transformed using the Cauchy theorem as follows:

$$
V_{ij}(\theta) = e^{-i\theta} \int_0^{\infty} \phi_i\left(r e^{-i\theta}\right) V(r) \phi_j\left(r e^{-i\theta}\right) , dr
$$

This transformation allows the potential to be evaluated using only its values at the mesh points along the real axis. It is important to note that, after performing the backward rotation, the basis functions oscillate much more strongly than before. Consequently, all matrix elements must be evaluated using the Gauss quadrature method with an increased number of mesh points.

In the standard method (θ = 0), the Gauss–Laguerre quadrature with ( $n_x$ ) points provides a good approximation of the integral ( $\int \phi_i(r) V(r) \phi_j(r),dr$ ). The localized nature of the basis functions ensures that this limited number of quadrature points is sufficient, typically resulting in a nearly diagonal matrix structure. However, under complex scaling (θ ≠ 0), the rotated basis functions ( $\phi_i(r e^{-i\theta})$ ) exhibit much stronger oscillations. As a result, the same ( $n_x$ )-point quadrature becomes insufficient to achieve accurate integration. To properly capture these oscillations, the number of quadrature points must be significantly increased (( $n_{\text{Gauss}} \gg n_x )$), and the standard Gauss–Laguerre quadrature should be employed without restricting the evaluation to the original mesh points along the real axis.

## Scaling the inhomogeneous term 

When applied the complex scaling the inhomogeneous term can be written as  
$$
[b(\theta)]  = 2*[V(\theta)] [R_{k_x, k_y, i_x, i_y}^{\alpha_3 \gets \alpha_1}] [\varphi(\theta)].
$$
with 
$$
\varphi_i(\theta) = \frac{\phi_d^{\alpha_1}(x_ie^{i\theta}) F^{\alpha_1}(y_ie^{i\theta})}{f_{i_x}(x_i) f_{i_y}(y_i)},
$$

> **Complex-scaling angle constraint (Lazauskas-Carbonell; HDR arXiv:1904.04675 Eq.2.143). This is the
> single most important practical point and the one most easily missed.**
> The inhomogeneous term $b(\theta)$ stays exponentially bound (and the scattered wave decays) only if
> $\theta$ is small enough. In the non-proper coordinate set, the point $y_b\gg0,\ x_b=0$ (where $V_b\ne0$)
> maps to $x_a=|s_{ab}|y_b,\ y_a=|c_{ab}|y_b$; the source decays as $e^{-x_a q_d\cos\theta}$ (deuteron bound
> state) but grows as $e^{+y_a q_{scatt}\sin\theta}$ (incoming wave), so it is bound only if
> $$ \tan\theta < \left|\frac{s_{ab}}{c_{ab}}\right|\frac{q_{deuteron}}{q_{scatt}}. $$
> swift uses **PHYSICAL (non-mass-scaled) Jacobi coordinates**, so $|s_{ab}/c_{ab}|$ is smaller than the
> mass-scaled $\sqrt3$ and $\theta_{max}$ is correspondingly smaller than a naive estimate. For n-d at
> 14.1 MeV, $\theta=3$–$4^\circ$ works (Rimas Table 5.1: $\theta=3^\circ\to\delta=105.0^\circ,\ \eta=0.456$);
> $\theta=10^\circ$ is too large and the amplitude **diverges with the y-box** ($\eta$: 0.57 at $y_{max}{=}60$
> → 5549 at $y_{max}{=}120$). **If the extraction blows up with $y_{max}$, check this bound FIRST** — it is
> not an operator, normalization, or basis bug.

## Scattering amplitude

The amplitude is the Lazauskas-Carbonell Green-theorem integral relation (HDR arXiv:1904.04675
Eq.2.117-2.118). The crucial step is to **separate the Born term** (the two-incoming-wave piece) and
evaluate it WITHOUT complex scaling, because that term carries the fastest $\theta$-divergence:
$$
f(k) = -\frac{1}{E_{cm}}\Big[\, e^{2i\theta}\,\big\langle \phi_d F \,\big|\, V_{23}+V_{31} \,\big|\, \psi_3^{sc}\big\rangle_{CS}
\;+\; \big\langle \phi_d F \,\big|\, V_{23}+V_{31} \,\big|\, \psi_3^{in}\big\rangle_{\theta=0,\ \text{Born, no CS}} \,\Big],
$$
then the collision matrix $U=2ik_d f+1$ (below) and the channel-spin / Blatt-Biedenharn analysis.

> **Validated recipe (2026-06-18, after R. Lazauskas's reply; this SUPERSEDES the earlier debugging
> notes that were partly wrong and cost a lot of time).**
>
> - **Bra = the REGULAR incoming** $\phi_d F$ (NOT the outgoing Hankel / $e^{-iq_n y e^{i\theta}}$ form —
>   an earlier note had this backwards, which was a dead end). Use the **bilinear c-product**
>   (transpose, no conjugation) under CS: the rotated Hamiltonian is complex-symmetric, not Hermitian.
> - **Born term on the real axis ($\theta=0$):** $\langle\phi_d F|V_{23}+V_{31}|\psi_3^{in}\rangle$ is
>   built from un-rotated matrices. It is mesh-stable and is the piece that diverges fastest with
>   $\theta$, so it must NOT be placed under CS. Only the scattered term
>   $\langle\phi_d F|\cdots|\psi_3^{sc}\rangle$ gets CS (the scattered wave decays, so it converges).
>   Lumping $\psi^{in}+\psi^{sc}$ together under CS (the un-separated Eq.2.117) is what diverges.
> - **CS contour Jacobian $=e^{2i\theta}$** for the 3-body scattered term: one $e^{i\theta}$ per rotated
>   radial integral ($x$ and $y$). swift's $V$ matrix already carries the Hamiltonian's $e^{-i\theta}$ on
>   the $x$-contour, so the explicit factor cancels it on $x$ and supplies the **missing $y$-contour
>   factor** (the real overlap $N_y$ carries no contour factor). The 2-body case (one radial integral)
>   needs $e^{i\theta}$.
> - **Operator** $[V_{23}+V_{31}]=\mathcal R\,V_{12}$ ($\mathcal R=\text{Rxy}=P^++P^-$), realized in swift
>   as $V\cdot\text{Rxy}$ acting on the appropriate component.
> - **Deuteron c-norm $1/C_n$** ($C_n=\phi_d^{\mathsf T}B\phi_d=e^{2i\gamma}|C_n|$, $|C_n|\approx1$
>   $\theta$-stable) removes the eigenvector's arbitrary global phase $e^{i\gamma}$; pre-normalize
>   $\phi_d\to\phi_d/\sqrt{C_n}$ so that $\phi_d^{\mathsf T}B\phi_d=1$.
> - **back-rotation is correct and numerically MORE stable** than forward coordinate rotation (R.
>   Lazauskas + COLOSS): the real Lagrange basis stays well-conditioned, whereas forward rotation would
>   evaluate basis functions at complex arguments (ill-conditioned). swift already uses back-rotation.
>
> **Status (swift.jl, $\theta=3^\circ$, $e^{2i\theta}$; CONVERGED 2026-06-18):** $\delta\approx104^\circ$
> (benchmark $105.49^\circ$, within $\sim2^\circ$; was stuck at $77^\circ$ or diverging for the whole
> saga). $\eta\approx0.35$ vs benchmark $0.4649$ ($\sim25\%$ low). This residual survives every
> convergence and structure test:
> - **Mesh-converged.** At $n_{ch}=2$, $\eta\to0.348$ for $n_y\gtrsim120$ at $y_{max}=120$, and stays
>   $0.34$–$0.36$ for $y_{max}=60$–$180$. The earlier "$\eta$ drifts with $y_{max}$ / $\eta=0.44$" readings
>   were coarse-$n_y$ under-resolution artifacts, now resolved.
> - **Channel-INVARIANT.** $l_{max}$ 0→4 / $n_{ch}$ 2→26 give byte-identical $f_{sc}$ at fixed mesh. For
>   $J_{tot}=1/2$ the spectator $\lambda$ is capped by $J_{12}\otimes J_3=1/2$ ($\lambda=4$ cannot couple
>   for the deuteron $J_{12}=1$), and MT's S-wave-only $V$ makes the extra $j_{2b}$ channels decoupled free
>   spectators (the channel coupling in $A=E B-T-V(I+\text{Rxy})$ runs only through $V\cdot\text{Rxy}$). So
>   the gap is NOT model-space truncation.
> - **NOT channel-spin recoupling / D-state.** MT is central $\Rightarrow$ the deuteron is pure $^3S_1$
>   (0% D-state), so the bi-conjugate / Blatt-Biedenharn-recoupling hypotheses are moot here.
> - **2-body machinery validated.** The scalar analog (`test_2body_cs_1S0.jl`, MT $^1S_0$) reproduces
>   $\eta=0.999$ (unitarity) and $\delta=63.2^\circ$, uniquely pinning the per-radial CS Jacobian to
>   $e^{+1\theta}$. Amplitude structure variants (conjugated bra, bra/ket swap, $(1+\text{Rxy})\psi$ full
>   wavefunction, plain-$V$ operator) all ruled out: the baseline above IS the paper formula.
>
> The converged amplitude needs $\times1.075$ magnitude $+2.7^\circ$ phase to hit the benchmark: a FIXED
> $\sim7.5\%$ normalization $+$ small phase, not convergence/channels. Open leads: (a) swift's MT
> parametrization vs the exact potential behind PRC 84 Tab.III ($E_d=-2.2295$ MeV here); (b) a subtle
> amplitude prefactor. Diagnostic of record: `swift/test_3body_greens.jl`.
>
> **Why the earlier notes misled (for the record):** the un-separated Eq.2.117 form above lumps
> $\psi^{in}+\psi^{sc}$ under CS (diverges); the implementation note used the outgoing-exp bra and only a
> single $e^{i\theta}$ Jacobian (2-body), and never stated the angle constraint. The two things that
> actually fixed the PHASE (small $\theta$ + Born-on-real-axis) came from Lazauskas's reply. The
> "channel-spin recoupling / $y$-mesh" remaining-work note (now corrected) was wrong: $\eta$ is
> channel-invariant and mesh-converged at $0.35$.

The scattering amplitude can also be written as $f^{\alpha_0,\alpha_0'}(k)$, where $\alpha_0$ indexes the channel in which the deuteron remains in its ground state. The scattering matrix can then be computed through 
$$
U^{\alpha_0,\alpha_0'}(k) = 2ik_d f^{\alpha_0,\alpha_0'}(k) + 1 
$$

Following Seyler (Nucl. Phys. A **124**, 253-272, 1969), we use the channel spin representation, which is based on the channel total spin $\mathbb{S}$:
$$
\mathbb{S} = J_{d} + s_3 
$$
This representation requires the following recoupling transformation:
$$
U^{J}_{\lambda'_3\mathbb{S}',\lambda_3\mathbb{S}} = \sum_{J_3,J_3'}\hat{J}_3\hat{J}_3'\hat{\mathbb{S}}\hat{\mathbb{S}}' (-)^{2J-J_3-J_3'} \left\{
\begin{array}{ccc}
\lambda_3' & \tfrac{1}{2} & J'_3 \\
J_{12} & J & \mathbb{S}'
\end{array}
\right\}\left\{
\begin{array}{ccc}
\lambda_3 & \tfrac{1}{2} & J_3 \\
J_{12} & J & \mathbb{S}
\end{array}
\right\} U_{\lambda_3'J_3',\lambda_3J_3}^J
$$
where $\hat{J} = \sqrt{2J+1}$ denotes the dimension factor.

For $Nd$ scattering, the scattering matrix $U$ need not be diagonal in channel spin. Thus, for a particular value of total angular momentum $J$ and parity $\pi$, it is necessary to consider two $3\times 3$ matrices:
$$
U^{J \pi}=\left[\begin{array}{lll}
U_{J \mp \frac{3}{2}, \frac{3}{2}; J \mp \frac{3}{2}, \frac{3}{2}}^J & U_{J \mp \frac{3}{2}, \frac{3}{2}; J \pm \frac{1}{2}, \frac{1}{2}}^J  & U_{J \mp \frac{3}{2}, \frac{3}{2}; J \pm \frac{1}{2}, \frac{3}{2}}^J \\
U_{J \pm \frac{1}{2}, \frac{1}{2}; J \mp \frac{3}{2}, \frac{3}{2}}^J & U_{J \pm \frac{1}{2}, \frac{1}{2}; J \pm \frac{1}{2}, \frac{1}{2}}^J & U_{J \pm \frac{1}{2}, \frac{1}{2}; J \pm \frac{1}{2}, \frac{3}{2}}^J \\
U_{J \pm \frac{1}{2}, \frac{3}{2}; J \mp \frac{3}{2}, \frac{3}{2}}^J & U_{J \pm \frac{1}{2}, \frac{3}{2}; J \pm \frac{1}{2}, \frac{1}{2}}^J & U_{J \pm \frac{1}{2}, \frac{3}{2}; J \pm \frac{1}{2}, \frac{3}{2}}^J
\end{array}\right],
$$
with parity $\pi=(-)^{J\pm \frac{1}{2}}$.

The scattering matrix can be parameterized using the Blatt-Biedenharn representation:
$$
U^{J\pi} = (u^{J\pi})^T \exp(2i\delta^{J\pi}) (u^{J\pi})
$$
where $\delta^{J\pi}$ is a diagonal matrix of eigenphase shifts:
$$
\boldsymbol{\delta}^{J \pi}=\left[\begin{array}{ccc}
\delta_{J \mp \frac{3}{2}, \frac{3}{2}}^J & 0 & 0 \\
0 & \delta_{J \pm \frac{1}{2}, \frac{1}{2}}^J & 0 \\
0 & 0 & \delta_{J \pm \frac{1}{2}, \frac{3}{2}}^J
\end{array}\right]
$$

The eigenphase shifts are labeled with the $l$ and $s$ subscripts of the partial wave whose phase shift they would correspond to in the limit of no partial wave mixing. The real orthogonal matrix $\boldsymbol{u}^{J \pi}$ is parameterized by three successive rotation matrices (Blatt-Biedenharn rotations):
$$
\begin{gathered}
\boldsymbol{u}^{J \pi}=\boldsymbol{v}^{J \pi} \boldsymbol{w}^{J \pi} \boldsymbol{x}^{J \pi}, \\
\boldsymbol{v}^{J \pi}=\left[\begin{array}{ccc}
1 & 0 & 0 \\
0 & \cos \varepsilon^{J \pi} & \sin \varepsilon^{J \pi} \\
0 & -\sin \varepsilon^{J \pi} & \cos \varepsilon^{J \pi}
\end{array}\right], \\
\boldsymbol{w}^{J \pi}=\left[\begin{array}{ccc}
\cos \zeta^{J \pi} & 0 & \sin \zeta^{J \pi} \\
0 & 1 & 0 \\
-\sin \zeta^{J \pi} & 0 & \cos \zeta^{J \pi}
\end{array}\right], \\
\boldsymbol{x}^{J \pi}=\left[\begin{array}{ccc}
\cos \eta^{J \pi} & \sin \eta^{J \pi} & 0 \\
-\sin \eta^{J \pi} & \cos \eta^{J \pi} & 0 \\
0 & 0 & 1
\end{array}\right].
\end{gathered}
$$

The eigenphase shifts and mixing parameters are computed through the following procedure. First, diagonalize the collision matrix $\mathbf{U}^{J\pi}$ to find the complex eigenvalues $\lambda_k$ and the real eigenvectors. The eigenvalues are related to the eigenphase shifts by $\lambda_k = \exp(2i\delta_k)$. Therefore, each eigenphase shift $\delta_k$ is computed from the phase of the corresponding eigenvalue:
$$
\delta_k = \frac{1}{2} \arg(\lambda_k)
$$

The orthogonal mixing matrix $\mathbf{u}^{J\pi}$ is formed by the eigenvectors of $\mathbf{U}^{J\pi}$. The columns of $\mathbf{u}^{J\pi}$ are the real, normalized eigenvectors of $\mathbf{U}^{J\pi}$. Once the orthogonal matrix $\mathbf{u}^{J\pi}$ is determined, the mixing parameters $\varepsilon$, $\zeta$, and $\eta$ are extracted by decomposing $\mathbf{u}^{J\pi}$ into three successive rotations:
$$
\boldsymbol{u}^{J \pi}=\boldsymbol{v}^{J \pi} \boldsymbol{w}^{J \pi} \boldsymbol{x}^{J \pi}
$$

For the $3 \times 3$ matrix case, the parameters are extracted sequentially. The element $\mathbf{u}_{13}^{J\pi}$ equals $\sin \zeta^{J\pi}$:
$$
\zeta^{J\pi} = \arcsin(\mathbf{u}_{13}^{J\pi})
$$
The element $\mathbf{u}_{11}^{J\pi}$ equals $\cos \eta^{J\pi} \cos \zeta^{J\pi}$. Knowing $\zeta$:
$$
\eta^{J\pi} = \arccos\left(\frac{\mathbf{u}_{11}^{J\pi}}{\cos \zeta^{J\pi}}\right)
$$
The element $\mathbf{u}_{23}^{J\pi}$ equals $\sin \varepsilon^{J\pi} \cos \zeta^{J\pi}$. Knowing $\zeta$:
$$
\varepsilon^{J\pi} = \arcsin\left(\frac{\mathbf{u}_{23}^{J\pi}}{\cos \zeta^{J\pi}}\right)
$$

This general procedure is applicable to the $3 \times 3$ case (e.g., $J^\pi = 5/2^+$).

For the $J=1/2$ total angular momentum state, the collision submatrices ($\mathbf{U}^{J\pi}$) are only $2 \times 2$. This is because not all partial wave states that exist in the $3 \times 3$ general case can couple to $J=1/2$ while maintaining non-negative orbital angular momentum quantum numbers. Since the matrix is $2 \times 2$, only two eigenphase shifts and one mixing parameter are needed for the phase shift analysis.

The $2 \times 2$ collision matrix is parameterized as:
$$
\mathbf{U}^{J\pi} = (\mathbf{u}^{J\pi})^{T}\exp(2i\boldsymbol{\delta}^{J\pi})(\mathbf{u}^{J\pi})
$$
where $\boldsymbol{\delta}^{J\pi} = \begin{pmatrix} \delta_1 & 0 \\ 0 & \delta_2 \end{pmatrix}$ contains the two eigenphase shifts, and $\mathbf{u}^{J\pi}$ is the $2 \times 2$ orthogonal mixing matrix, parameterized by a single mixing angle $\alpha$:
$$
\mathbf{u}^{J\pi} = \begin{pmatrix} \cos \alpha & \sin \alpha \\ -\sin \alpha & \cos \alpha \end{pmatrix}
$$

The mixing angle $\alpha$ is denoted as $\eta$ for $J^\pi=1/2^+$ and $\varepsilon$ for $J^\pi=1/2^-$. The eigenphase shifts are computed from the eigenvalues of $\mathbf{U}^{J\pi}$:
$$
\delta_{k} = \frac{1}{2} \arg(\lambda_k)
$$
where $\lambda_k$ are the two complex eigenvalues. The single mixing angle $\alpha$ is determined from the eigenvectors that form the columns of $\mathbf{u}^{J\pi} = \begin{pmatrix} u_{11} & u_{12} \\ u_{21} & u_{22} \end{pmatrix}$:
$$
\alpha = \arccos(u_{11}) = \arcsin(u_{12})
$$

The process is fundamentally the same as the $3 \times 3$ case: diagonalize the collision matrix to find the eigenphase shifts and the orthogonal mixing matrix, then decompose the mixing matrix to extract the mixing parameters.

The mixing parameters describe different types of partial wave coupling. The parameter $\varepsilon$ (epsilon) measures spin mixing without orbital angular momentum mixing between partial waves. It corresponds to the rotation matrix $\mathbf{v}^{J\pi}$, coupling states with the same orbital angular momentum ($l$) but different channel spins ($\mathbb{S}$). For example, it measures the mixing between states like $^4P_{1/2}$ ($l=1, \mathbb{S}=3/2$) and $^2P_{1/2}$ ($l=1, \mathbb{S}=1/2$). The parameter $\zeta$ (zeta) measures orbital angular momentum mixing without spin mixing. It corresponds to the rotation matrix $\mathbf{w}^{J\pi}$, coupling states with the same channel spin ($\mathbb{S}$) but different orbital angular momenta (typically $l$ and $l \pm 2$). This is analogous to the $\varepsilon$ mixing parameter used in nucleon-nucleon scattering. The parameter $\eta$ (eta) measures the mixing between partial waves that differ in both their channel spin and orbital angular momentum quantum numbers. It corresponds to the rotation matrix $\mathbf{x}^{J\pi}$ and captures the remaining type of partial wave coupling.

In summary, the eigenphase shifts ($\delta$) describe the strength of the interaction in the diagonalized (uncoupled) basis, while the mixing parameters ($\varepsilon, \zeta, \eta$) quantify the extent to which the physically observed states are mixtures of the theoretical partial wave states. Non-zero mixing parameters indicate that channel spin is not conserved in the scattering process, reflecting the complexity of three-body nuclear dynamics.





theta=8 

ymax=80    ny=60 xmax=20

J^π = 0.5^+:

  Eigenphase shifts:

    δ_1 = 27.0700° (0.472460 rad)

    δ_2 = -0.0397° (-0.000692 rad)

  Mixing parameters:

    η = 154.7959° (2.701698 rad)

ymax=60 ny=60 xmax=20

J^π = 0.5^+:

  Eigenphase shifts:

    δ_1 = 28.4198° (0.496019 rad)

    δ_2 = -0.0224° (-0.000391 rad)

  Mixing parameters:

    η = 154.4284° (2.695284 rad)

ymax=60 ny=70 xmax=20  nx=24
J^π = 0.5^+:

  Eigenphase shifts:

    δ_1 = 28.5778° (0.498777 rad)

    δ_2 = -0.0242° (-0.000423 rad)

  Mixing parameters:

    η = 154.4976° (2.696493 rad)



ymax=60 ny=70 xmax=30   nx=24
J^π = 0.5^+:

  Eigenphase shifts:

    δ_1 = 27.5107° (0.480152 rad)

    δ_2 = -1.3068° (-0.022809 rad)

  Mixing parameters:

    η = 177.7911° (3.103040 rad)