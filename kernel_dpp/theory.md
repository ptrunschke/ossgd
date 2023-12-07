# RKHS-optimal sampling

## Setting

Consider a RKHS $\mathcal{V}$ with kernel $k : \mathcal{X}\times\mathcal{X}\to\mathbb{R}$ and let the $d$-dimensional subspace $\mathcal{V}_d\subseteq\mathcal{V}$ be spanned by the $\mathcal{V}$-ONB $b_1,\ldots,b_d$.

Let $u\in L^2$ and a given sample $x_1, \ldots, x_n\in\mathcal{X}$ let $K_{ij} := k(x_i, x_j)$ then define
$$
    u_{d,n} := \argmin_{v\in\mathcal{V}_d} \|u - v\|_{n}
    \qquad\text{with}\qquad
    \|v\|_n := \boldsymbol{v}^\intercal K^+ \boldsymbol{v}
    \qquad\text{and}\qquad
    \boldsymbol{v}_i := v(x_i) .
$$
Defining further $u_{\star}\in\mathcal{V}$ as the $L^\infty$-best approximation of $u$ in $\mathcal{V}$ and $u_d\in\mathcal{V}_d$ as the $\mathcal{V}$-best approximation of $u_\star$ in $\mathcal{V}_d$, it has been shown that
$$
    \|u_\star - u_{d,n}\|_{\mathcal{V}}
    \le (1 + 2\mu) \|u_\star - u_d\|_{\mathcal{V}_d}  + (1 + 2\mu_\infty) \|u - u_\star\|_{L^\infty} .
$$
Defining $B_{li} := b_l(x_i)$, the constants $\mu$ and $\mu_\infty$ are given by
$$
    \mu
    := \max_{v\in\mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|v\|_{n}}
    := \lambda_{\mathrm{min}}(BK^+B^\intercal)^{-1}
    \qquad\text{and}\qquad
    \mu_\infty
    := \max_{v\in\mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|\boldsymbol{v}\|_{\infty}} .
$$

## Sampling
Assuming $\|u-u_\star\|_{L^\infty}$ to be negligible, we can optimise the points as to minimise the consant $\mu$.
As a surrogate for the maximisation of the smallest eigenvalue we propose to maximise the determinant.
Choosing $n=d$ sample points and conditioning on the event $\operatorname{ker}(K)=0$, we obtain
$$
    \det(BK^+ B^\intercal) = \frac{\det(B^\intercal B)}{\det(K)} .
$$
Now, assume that a sufficiently good rank-$R$ approximation of the Mercer decomposition
$$
    k(x, y) = \sum_{l=1}^\infty \sigma_l \phi_l(x) \phi_l(y) \approx \sum_{l=1}^R \psi_l(x)\psi_l(y)
$$
is given.
Then we can define $\Psi_{li} := \psi_l(x_i)$ and write
$$
    K_{ij} = k(x_i, x_j) = \sum_{l=1}^R \psi_l(x_i)\psi_l(x_j) = (\Psi^\intercal \Psi)_{ij} .
$$
Now recall that for any matrix $A = [a_1\ \tilde{A}_1]$, the Woodbury matrix identity implies
$$
    \det(A^\intercal A) = a_1^\intercal(I - \tilde{A}_1\tilde{A}_1^+)a_1 \cdot \det(\tilde{A}_1^\intercal \tilde{A}_1) .
$$
Recursive application of this relation yields
$$
    \det(BK^+ B^\intercal)
    = \frac{\det(B^\intercal B)}{\det(\Psi^\intercal\Psi)}
    = \prod_{i=1}^n \frac{\boldsymbol{b}_i^\intercal(I - \tilde{B}_i\tilde{B}_i^+) \boldsymbol{b}_i}{\boldsymbol{\psi}_i^\intercal (I - \tilde{\Psi}_i\tilde{\Psi}_i^+) \boldsymbol{\psi}_i}
    .
$$
Here $\tilde{B}_i \in \mathbb{R}^{d\times (n-i)}$ and $\tilde{\Psi}_i \in \mathbb{R}^{R\times (n-i)}$ are given by
$$
    (\tilde{B}_i)_{lj} := b_l(x_j)
    \qquad\text{and}\qquad
    (\tilde{\Psi}_i)_{lj} := \psi_l(x_j),
$$
respectively,
and $\boldsymbol{b}_i\in\mathbb{R}^d$ and $\boldsymbol{\psi}_i\in\mathbb{R}^R$ are defined by
$$
    (\boldsymbol{b}_i)_l := b_l(x_i)
    \qquad\text{and}\qquad
    (\boldsymbol{\psi}_i)_l := \psi_l(x_i),
$$
respectively.
Since $\tilde{B}_i$ and $\tilde\Psi_i$ only depend on the points $x_{j}$ for $j>i$, this demonstrates how we can recursively sample from the (unnormalised) density $\det(BK^+ B^\intercal)$ by sequentially sampling from the ratios
$$
    \frac{\boldsymbol{b}_i^\intercal(I - \tilde{B}_i\tilde{B}_i^+) \boldsymbol{b}_i}{\boldsymbol{\psi}_i^\intercal (I - \tilde{\Psi}_i\tilde{\Psi}_i^+) \boldsymbol{\psi}_i} .
$$
