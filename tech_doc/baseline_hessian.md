# Traditional Branch Hessian Formulation (Baseline)

To contrast with our revolutionary single-pass direct-fill approach, we document the traditional branch Hessian formulation used in MATPOWER and PYPOWER.

## 1. Lagrangian Term for Branch Limits

The inequality constraint for a branch flow limit (apparent power squared) is:
$$ h_l(\mathbf{x}) = |S_l|^2 - S_{max,l}^2 \le 0 $$
The corresponding Lagrangian term is:
$$ \mathcal{L}_{br} = \sum_{l=1}^{m} \mu_l \cdot h_l(\mathbf{x}) $$

## 2. Polar Jacobian Components (Wirtinger Form)

The apparent power at the "from" end of branch $l$ is $S_f = V_f \cdot I_f^*$, where $I_f = Y_{ff} V_f + Y_{ft} V_t$.
The first derivatives in polar coordinates $(\theta, V_m)$ involve the following complex sensitivity operators:

$$ \frac{\partial S_f}{\partial \theta} = j \operatorname{diag}(V_f) [ \operatorname{diag}(I_f)^* - \mathbf{Y}_{f} \operatorname{diag}(V)^* ] $$
$$ \frac{\partial S_f}{\partial V_m} = \operatorname{diag}(V_f) \mathbf{Y}_{f} \operatorname{diag}(\hat{V})^* + \operatorname{diag}(I_f) \operatorname{diag}(\hat{V})^* $$

## 3. The $|S|^2$ Chain Rule

The Hessian of $|S|^2$ is derived using the identity $|S|^2 = S \cdot S^*$:
$$ \nabla^2 (|S|^2) = \nabla^2 (S \cdot S^*) = 2 \operatorname{Re} \left\{ (S^*) \nabla^2 S + (\nabla S)^H (\nabla S) \right\} $$

Substituting the Lagrangian multiplier $\mu$, the traditional Hessian contribution for branch constraints is:
$$ \mathbf{H}_{br} = 2 \sum_{l=1}^{m} \mu_l \operatorname{Re} \left\{ S_l^* \nabla^2 S_l + (\nabla S_l)^H (\nabla S_l) \right\} $$

## 4. Computational Steps in the Traditional Path

The baseline path (`d2Sbr_dv2.m` or our `d2Sbr_dv2.rs`) executes the following discrete steps:

1.  **Intermediate Vector Computation**: Calculate $S_l$ and $I_l$ for all branches.
2.  **Jacobian Matrix Materialization**: Explicitly form the sparse Jacobian matrices $\frac{\partial S}{\partial \theta}$ and $\frac{\partial S}{\partial V_m}$.
3.  **Complex Matrix-Matrix Product**: Evaluate $(\nabla S)^H (\nabla S)$. This is a sparse SpGEMM operation $(\text{nb} \times \text{nl}) \times (\text{nl} \times \text{nb})$ which results in a dense-like structure if not handled carefully.
4.  **Second Derivative Computation**: Evaluate the sparse tensor-product $S^* \nabla^2 S$.
5.  **Multi-Matrix Summation**: Sum the resulting sub-blocks ($H_{aa}, H_{av}, H_{va}, H_{vv}$) using sparse addition (`csc_add`).
6.  **Scattered Assembly**: Copy the summed blocks into the global Lagrangian Hessian $L_{xx}$ at scattered indices.

### Why this is the "Antithesis":
- **Structural Fragmentation**: Each step above creates new sparse matrix structures, requiring repeated symbolic analysis or expensive binary searches.
- **Redundant Complex Math**: Millions of complex multiplications are performed to generate intermediate matrices that are only used once then discarded.
- **Memory Pressure**: The peak memory allocation spikes during the summation of these intermediate blocks.
