# V3 Branch Hessian Fusion: Mathematical Blueprint

This document details the step-by-step mathematical decomposition of the branch flow limit Hessian, preparing it for the "Single-Pass FMA" implementation in Rust. We adhere to the Wirtinger calculus pull-back paradigm outlined in `infer.md` and `doc2.md`.

## 1. The Constraint and its Differential

For a branch $l$ connecting bus $i$ (from) and bus $k$ (to), the apparent power squared at the "from" end is:
$$ h_l = S_f S_f^* $$
Where $S_f = V_i I_f^*$ and the complex branch current is $I_f = Y_{ff} V_i + Y_{ft} V_k$.

The second differential is:
$$ d^2 h_l = 2 \text{Re} \left\{ dS_f^* dS_f + S_f^* d^2 S_f \right\} $$

## 2. Unpacking the Complex Components

We work in the local $2 \times 1$ complex voltage space $\mathbf{V} = [V_i, V_k]^T$.

### A. The First Derivative (Jacobian) $J_{S}$
$J_S$ is a $1 \times 4$ row vector representing the complex sensitivity w.r.t $[V_i, V_k, V_i^*, V_k^*]$.
Taking the Wirtinger derivatives of $S_f = V_i (Y_{ff}^* V_i^* + Y_{ft}^* V_k^*)$:

*   $\frac{\partial S_f}{\partial V_i} = I_f^*$
*   $\frac{\partial S_f}{\partial V_k} = 0$
*   $\frac{\partial S_f}{\partial V_i^*} = V_i Y_{ff}^*$
*   $\frac{\partial S_f}{\partial V_k^*} = V_i Y_{ft}^*$

So, $J_S = \begin{bmatrix} I_f^* & 0 & V_i Y_{ff}^* & V_i Y_{ft}^* \end{bmatrix}$.

### B. The Second Derivative $\mathcal{H}_S$
$\mathcal{H}_S$ is the $4 \times 4$ complex Hessian of the power injection $S_f$.
Because $S_f$ only has cross terms like $V_i V_i^*$ and $V_i V_k^*$, all pure second derivatives ($\frac{\partial^2}{\partial V \partial V}$ and $\frac{\partial^2}{\partial V^* \partial V^*}$) are strictly zero.

The only non-zero block is the mixed derivative $\frac{\partial^2 S_f}{\partial V^* \partial V}$:
*   $\frac{\partial}{\partial V_i^*} \left( \frac{\partial S_f}{\partial V_i} \right) = Y_{ff}^*$
*   $\frac{\partial}{\partial V_k^*} \left( \frac{\partial S_f}{\partial V_i} \right) = Y_{ft}^*$

Thus, the full $4 \times 4$ matrix is a sparse constant network operator:
$$
\mathcal{H}_S = \begin{bmatrix} 
0 & 0 & Y_{ff}^* & Y_{ft}^* \\ 
0 & 0 & 0 & 0 \\ 
Y_{ff}^* & 0 & 0 & 0 \\ 
Y_{ft}^* & 0 & 0 & 0 
\end{bmatrix}
$$

## 3. The Complex Hessian of $|S_f|^2$
Applying the components back to the differential $d^2 h_l$, the $4 \times 4$ complex Hessian for the limit is:
$$ \mathcal{H}_{h, l} = 2 J_S^H J_S + 2 \text{Re} \{ S_f^* \cdot \mathcal{H}_S \} $$

*Note on $2 \text{Re}\{ \cdot \}$ for matrices: For a matrix $A$, $2 \text{Re}\{ A \}$ means $A + A^H$ in the context of creating a Hermitian form.*

Let's denote the constant scaled part as $C = S_f^* \mathcal{H}_S$.
$$
C = \begin{bmatrix} 
0 & 0 & S_f^* Y_{ff}^* & S_f^* Y_{ft}^* \\ 
0 & 0 & 0 & 0 \\ 
S_f^* Y_{ff}^* & 0 & 0 & 0 \\ 
S_f^* Y_{ft}^* & 0 & 0 & 0 
\end{bmatrix}
$$
The Hermitian sum $C + C^H$:
$$
C + C^H = \begin{bmatrix} 
0 & 0 & S_f^* Y_{ff}^* & S_f^* Y_{ft}^* \\ 
0 & 0 & 0 & 0 \\ 
S_f Y_{ff} & 0 & 0 & 0 \\ 
S_f Y_{ft} & 0 & 0 & 0 
\end{bmatrix}
$$

So, the total complex Hessian $\mathcal{H}_{h,l}$ is simply the rank-1 update plus this highly sparse structure.

## 4. The Polar Pull-back
To map this $4 \times 4$ complex Hessian to the $4 \times 4$ real polar Hessian for variables $[\theta_i, \theta_k, |V_i|, |V_k|]$, we apply the rotation operator $M_{\mathbb{C}}$:

$$
M_{\mathbb{C}} = \begin{bmatrix}
j V_i & 0 & V_i / |V_i| & 0 \\
0 & j V_k & 0 & V_k / |V_k| \\
-j V_i^* & 0 & V_i^* / |V_i| & 0 \\
0 & -j V_k^* & 0 & V_k^* / |V_k|
\end{bmatrix}
$$

The final real Hessian contribution to $L_{xx}$ is:
$$ H_{polar, l} = \text{Re} \left\{ M_{\mathbb{C}}^H \cdot \mathcal{H}_{h,l} \cdot M_{\mathbb{C}} \right\} $$

### Implementation Step-by-Step
To ensure mathematical correctness before optimization, we will:
1.  **Extract Local Variables**: For each branch, get $V_i, V_k, Y_{ff}, Y_{ft}, Y_{tf}, Y_{tt}$.
2.  **Compute Currents and Powers**: $I_f, S_f$ and $I_t, S_t$.
3.  **Construct $J_S$ and $\mathcal{H}_S$**: As defined in Section 2.
4.  **Assemble $\mathcal{H}_{h}$**: Use nalgebra `Matrix4<Complex64>` to literally evaluate $2 J_S^H J_S + (C + C^H)$.
5.  **Apply $M_{\mathbb{C}}$**: Construct the $4 \times 4$ matrix $M_{\mathbb{C}}$ and evaluate $Re\{ M_{\mathbb{C}}^H \mathcal{H}_h M_{\mathbb{C}} \}$.
6.  **Accumulate**: Add the resulting $4 \times 4$ real matrix directly to the global $L_{xx}$ using the `br_to_lxx` index cache.

Once this un-optimized, matrix-based formulation passes the `test_compare_hessian` baseline check, we can safely unroll it into pure scalar FMAs.