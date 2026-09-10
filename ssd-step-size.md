Where the denominator comes from
The step size minimises the expected loss after one step. For any direction $z$,

$$\mathbf{E}[\mathcal{L}(\theta+\alpha z)] = \mathcal{L}(\theta) + \alpha\,\nabla\mathcal{L}^\top\mathbf{E}[z] + \tfrac{\alpha^2}{2}\,\mathbf{E}[z^\top Q z]$$

which is a scalar quadratic in $\alpha$, minimised at $\alpha_\ast = -\nabla\mathcal{L}^\top\mathbf{E}[z] \,/\, \mathbf{E}[z^\top Qz]$. With $z=-s$, $s=\operatorname{sign}(g)$, the two minus signs cancel in the numerator and square away in the denominator:

$$\alpha_\ast = \frac{\nabla\mathcal{L}^\top\mathbf{E}[s]}{\mathbf{E}[s^\top Q s]}$$

So the denominator is the curvature felt along the sign direction. Note $|s|^2 = d$ always — sign descent picks a corner of the hypercube, so the denominator is $d$ times the average curvature over the directions the sign vector can take.

$Q$ comes out of the expectation, not out of the problem
$s^\top Qs$ is a scalar, so expand it in coordinates:

$$s^\top Q s = \sum_{i,j} q_{ij}, s_i s_j \quad\Longrightarrow\quad \mathbf{E}[s^\top Q s] = \sum_{i,j} q_{ij}, \mathbf{E}[s_i s_j] = \langle Q, M\rangle_F, \qquad M := \mathbf{E}[ss^\top]$$

$Q$ is deterministic, so it factors out as a fixed weight matrix; the only unknown left is $M$, the matrix of expected pairwise sign agreements. Equivalently $\mathbf{E}[s^\top Qs] = \operatorname{tr}(Q,\mathbf{E}[ss^\top])$ by the trace trick. That is exactly sqv.ipynb:86, np.sum(Q * M) — elementwise product then sum is the Frobenius inner product, not a matrix product.

So $Q$ appears in two places, which is the crux of your question:

Outside the expectation, as the weights $q_{ij}$ — line 86.
Inside, through the distribution of $s$ — line 83. The sampled gradient is $g = Q(\theta - x^\ast - \varepsilon) = \nabla\mathcal{L} - Q\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,\nu^2 I)$, so $\operatorname{cov}[g] = \nu^2 QQ$. $Q$ is what correlates the signs across coordinates. (rng.normal(...) @ Q is $(Q\varepsilon)^\top$ row-wise, since $Q$ is symmetric.)
Dropping $Q$ from either place would be wrong in a different way. Drop it from (1) and you get the wrong quadratic form; drop it from (2) and you get independent coordinates, i.e. you assume away the whole problem.

Why $M$ needs Monte Carlo
$s_i\in{\pm1}$, so $M_{ij} = \mathbf{P}[s_is_j{=}1] - \mathbf{P}[s_is_j{=}{-}1] = 2,\mathbf{P}[s_is_j{=}1]-1$. That probability is the mass of a bivariate Gaussian with non-zero mean in two opposite orthants — no elementary closed form. (Zero-mean has one, Sheppard's $1-\frac{2}{\pi}\arccos\varrho_{ij}$; the non-central case does not.)

The special case that does close: if $Q$ is diagonal, the $g_i$ are independent, so for $i\neq j$, $M_{ij} = (2\rho_i-1)(2\rho_j-1)\operatorname{sign}(\nabla\mathcal{L}_i)\operatorname{sign}(\nabla\mathcal{L}_j)$ — the same $\operatorname{erf}$ terms as the numerator. The code doesn't special-case this; it doesn't need to, see below.

The estimator is then just the sample second moment: S is $n_{mc}\times d$ of $\pm 1$, one row per sampled gradient, and S.T @ S / n_mc is $\frac{1}{n}\sum_k s^{(k)}s^{(k)\top} \approx \mathbf{E}[ss^\top]$. Cheap — one $d\times d$ Gram matrix per step. Measured stability: 0.003% relative std over 12 MC seeds at $n_{mc}=2000$ (the docstring's "~0.1%" is conservative).

Two notes on the code
np.fill_diagonal(M, 1.0) on line 85 is a no-op. Every entry of S is $\pm1$, so $(S^\top S){ii} = \sum_k (S{ki})^2 = n_{mc}$ identically — the diagonal is already exactly 1 before the call. I measured max|diag(M) - 1| = 0 exactly, both axis-aligned and rotated. The docstring's conclusion still holds — the estimate is exact for axis-aligned $Q$, because when $Q$ is diagonal only $M_{ii}$ is ever touched by $\langle Q,M\rangle$, and those are exact by construction — but it holds regardless of that line, and the "reduces variance elsewhere" clause is not doing anything. Worth either deleting the line or rewording the docstring; it currently credits the line with a property the estimator has on its own.

How loose the paper's bound actually is. §B.2 bounds $M_{ij}\le 1$, giving $\mathbf{E}[s^\top Qs] \le \sum_{ij}|q_{ij}| = \operatorname{tr}(Q)/p_{\text{diag}}(Q)$ — that is where the $\sum_i\lambda_i$ and $p_{\text{diag}}$ in the $\mathcal{I}_\textsc{ssd}$ lower bound come from. It assumes every pair of signs agrees perfectly. On the ill-conditioned instances at $\nu=0.1$:

$p_{\text{diag}}$	$\mathbf{E}[s^\top Qs]$ (MC)	bound	ratio
axis-aligned	1.000	573.7	573.7	1.0×
rotated	0.044	3934	13070	3.3×
Tight when $Q$ is diagonal (the bound is the answer), 3.3× too large once rotated — so using it as a step size would run SSD at $\alpha_\ast/3.3$ on exactly the problems where SSD is already struggling. Fine as a bound on $\mathcal{I}$, unusable as a step size.

For contrast, SGD needs no MC because $z=-g$ stays Gaussian: $\mathbf{E}[g^\top Qg] = \nabla\mathcal{L}^\top Q\nabla\mathcal{L} + \operatorname{tr}(Q\operatorname{cov}[g]) = \nabla\mathcal{L}^\top Q\nabla\mathcal{L} + \nu^2\sum_i\lambda_i^3$. Taking the sign is what destroys that.