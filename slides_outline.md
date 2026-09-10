# Dissecting Adam — Reproducibility Report

Slide outline. Balles & Hennig, ICML 2018 (arXiv:1705.07774).

---

## 1. Paper

**Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients.**

Claim: Adam = sign descent + variance adaptation. The sign is the dominant part.

Reproduced here: the sQP toy problem (Fig. 2) and Fashion-MNIST (P1).

---

## 2. The rewrite

Adam:

$$\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t}+\varepsilon}$$

Drop $\varepsilon$, factor out the sign:

$$\frac{m_t}{\sqrt{v_t}} = \sqrt{\frac{1}{1+\hat\eta_t^2}} \odot \operatorname{sign}(m_t),
\qquad
\hat\eta_{t,i}^2 = \frac{v_{t,i}-m_{t,i}^2}{m_{t,i}^2} \approx \frac{\sigma_{t,i}^2}{\nabla\mathcal{L}_{t,i}^2}$$

- Direction: sign of $m_t$.
- Magnitude: $\gamma_{t,i} = (1+\hat\eta_{t,i}^2)^{-1/2}$ — shrinks high relative-variance coordinates.

---

## 3. Four methods

|  | no variance adaptation | variance adaptation |
|---|---|---|
| **gradient** | M-SGD | M-SVAG |
| **sign** | M-SSD | Adam |

Two aspects, four recombinations. Isolate each by comparing across the grid.

---

## 4. Scope

Done:
- Fig. 2 — SGD vs. SSD on stochastic QPs, optimal local step sizes.
- P1 — Fashion-MNIST CNN, all four optimisers (M-SVAG reimplemented).

Not done: P2 CIFAR-10, P3 CIFAR-100, P4 LSTM. No step-size search. 1 seed, not 10.

---

## 5. Experiment 1 — model problem

$$\ell(\theta;x)=\tfrac12(\theta-x)^\top Q(\theta-x), \qquad x\sim\mathcal{N}(x^\ast,\nu^2 I)$$

$$\nabla\mathcal{L}(\theta)=Q(\theta-x^\ast), \qquad g(\theta)\sim\mathcal{N}(\nabla\mathcal{L},\,\nu^2 QQ)$$

$d=100$, 100 steps. $Q=\Lambda$ (axis-aligned) or $R\Lambda R^\top$, $R\sim$ Haar on $SO(100)$.

- well-conditioned: $\lambda_i \sim U[0.1,1.1]$
- ill-conditioned: 90% from $U[0,1]$, 10% from $U[30,60]$

Noise $\nu \in \{0,\,0.1,\,4.0\}$.

---

## 6. What the theory predicts

Expected improvement at the optimal local step size, $\mathcal{I}=\dfrac{(\nabla\mathcal{L}^\top\mathbf{E}[z])^2}{2\,\mathbf{E}[z^\top Q z]}$:

$$\mathcal{I}_{\text{SGD}} = \frac12 \frac{(\nabla\mathcal{L}^\top\nabla\mathcal{L})^2}{\nabla\mathcal{L}^\top Q\nabla\mathcal{L} + \nu^2\sum_i \lambda_i^3}$$

$$\mathcal{I}_{\text{SSD}} \geq \frac12 \frac{\big(\sum_i (2\rho_i-1)|\nabla\mathcal{L}_i|\big)^2}{\sum_i \lambda_i}\, p_{\text{diag}}(Q)$$

- $\nu^2\sum\lambda_i^3$: noise and spectrum interact — hurts SGD only.
- $p_{\text{diag}}(Q)$: 1 if diagonal, $\approx 1.57/d$ under random rotation — hurts SSD only.

Prediction: sign wins on noisy, ill-conditioned, axis-aligned problems.

---

## 7. Gap: the SSD step size

$\alpha_\ast = \dfrac{\nabla\mathcal{L}^\top \mathbf{E}[s]}{\mathbf{E}[s^\top Q s]}$, $s=\operatorname{sign}(g)$.

Numerator is closed form:

$$\nabla\mathcal{L}^\top\mathbf{E}[s]=\sum_i (2\rho_i-1)|\nabla\mathcal{L}_i|,
\qquad 2\rho_i-1=\operatorname{erf}\!\Big(\frac{|\nabla\mathcal{L}_i|}{\sqrt2\,\sigma_i}\Big)$$

Denominator needs $\mathbf{E}[s_i s_j]$ for correlated Gaussians with non-zero mean — orthant probabilities, no elementary form. §B.2 only bounds it; the bound is too loose to use as a step size.

Resolution: Monte Carlo $\mathbf{E}[ss^\top]$, $n=2000$, diagonal pinned to $1$. Exact for diagonal $Q$; $\alpha_\ast$ stable to ~0.1% across seeds.


SPEAKER NOTES — §7

WHY THE STEP SIZE MATTERS AT ALL
Fig. 2 runs each method at its own best local step size, so step-size tuning
is taken out of the comparison. If the SSD step is wrong, SSD is handicapped
(or helped) and the comparison stops being fair.

WHERE alpha* COMES FROM
E[L(theta - alpha s)] = L(theta) - alpha gradL^T E[s] + (alpha^2/2) E[s^T Q s].
That is a quadratic in alpha, so alpha* = (linear coeff) / (quadratic coeff).
The denominator is the expected curvature along the sign direction.

WHY THE NUMERATOR CLOSES
It is linear in s, so it only needs E[s_i], one coordinate at a time.
g_i ~ N(gradL_i, sigma_i^2), so P(g_i has the right sign) = Phi(|gradL_i|/sigma_i) = rho_i.
That is a 1-D normal CDF, and 2*Phi(x) - 1 = erf(x/sqrt2).

WHY THE DENOMINATOR DOESN'T
- Expand: E[s^T Q s] = sum_ij q_ij E[s_i s_j]. Q is a fixed weight matrix, so
  the only unknown is M = E[s s^T], the pairwise sign agreements.
- s_i s_j = +1 exactly when g_i and g_j have the same sign, so
  E[s_i s_j] = 2 P(same sign) - 1.
- P(same sign) = mass of a 2-D Gaussian in the (+,+) and (-,-) quadrants. That
  Gaussian has (a) a NON-ZERO MEAN (gradL_i, gradL_j) and (b) CORRELATION between
  the coordinates, because cov[g] = nu^2 QQ, which is off-diagonal once Q is
  rotated.
- (a) alone (diagonal Q => independent coords): it factorises into a product
  of the numerator's erf terms. Closed form.
- (b) alone (zero mean): Sheppard's formula, 1 - (2/pi) arccos(corr). After
  whitening, the quadrant becomes a wedge with its tip AT the mean, so by
  rotational symmetry the mass is just angle/2pi.
- (a)+(b) together: the wedge tip is no longer at the mean, the symmetry argument
  fails, and what's left is the bivariate normal CDF Phi_2(a, b; corr)
  (equivalently Owen's T function), an integral that does not reduce to erf.

If someone objects "erf isn't closed form either": fair. The precise claim is
that the numerator reduces to 1-D normal CDFs, which we accept as closed form,
and the denominator needs 2-D normal CDFs, which do not reduce to 1-D ones.
(Also, "orthant probabilities" on the slide really means 2-D QUADRANT
probabilities. s^T Q s is quadratic, so only pairs ever appear, never a
d-dimensional orthant.)

"NO CLOSED FORM" != "CAN'T COMPUTE"
Phi_2 is in scipy, and there are only d(d-1)/2 = 4950 pairs per step. I checked
this on ill-cond. rotated, nu = 0.1, theta = 0:
    exact (Phi_2): 3658.97    MC (n=2000): 3658.75    paper's bound: 12463
The exact computation took ~0.3 s. So MC was the simpler choice, not a forced
one. If asked, say it could be swapped for the exact computation at no real cost.

THE PAPER'S BOUND (§B.2)
It sets every M_ij <= 1, i.e. assumes every pair of signs always agrees, which
gives sum_ij |q_ij| = tr(Q) / p_diag(Q). For diagonal Q only the M_ii = 1 terms
enter, so the bound is exact. Rotated, it's ~3.4x too big (12463 vs 3659 above),
which would run SSD at about a third of its best step on exactly the panels
where SSD is already struggling. Fine for bounding I_SSD, but not usable as a
step size.

CAVEATS ON THE SLIDE WORDING (in case someone reads the code)
- "Diagonal pinned to 1" does nothing. Every entry of S is +-1, so
  diag(S^T S)/n = 1 exactly before the fill_diagonal call. Exactness for
  diagonal Q comes from <Q, M> only reading M_ii, and those are exact by
  construction.
- "~0.1%" is conservative: the measured relative std of alpha* over 12 MC
  seeds is 0.003% (see ssd-step-size.md).
- nu = 0 is special-cased: sign(g) is deterministic, so there's no MC.


---

## 8. Gap: unstated initialisation

$\theta_0$ and $x^\ast$ are not given in the paper. Both matter.

- $x^\ast \propto \mathbf{1}$ with $\theta_0=0$: the error lies along the sign direction, so axis-aligned SSD solves the problem in one line-searched step. Artefact.
- Offset scale sets the dynamic range above the noise floor. Small offset ⇒ both methods start at the floor ⇒ high-noise column is uninformative.

Used: $\theta_0=0$, $x^\ast\sim U[-10,10]^{100}$. Conclusions hold for any scale $\gtrsim 10$.


SPEAKER NOTES — §8

ONLY theta0 - x* MATTERS
The loss depends only on theta - x*, and the noise is additive and doesn't
depend on theta. So fixing theta0 = 0 loses nothing. The real choice is the
initial error e0 = theta0 - x*: its DIRECTION (bullet 1) and its SIZE (bullet 2).

BULLET 1: THE x* ∝ 1 ARTEFACT
- e0 = -c*1 with diagonal Q gives gradL = -c*lambda. Every entry has the same
  sign, so sign(gradL) = -1 and the SSD step is alpha*1. The line-searched
  alpha* = c lands exactly on x*.
- Checked at nu = 0 on both axis-aligned problems (c = 1 and c = 10): SSD's loss
  is exactly 0 after step 1. SGD after the same step is still at 3.2
  (well-cond.) and 26 (ill-cond.), from 30 and 273.
- More generally, the artefact appears whenever e0 is a scaled SIGN VECTOR
  (same magnitude in every coordinate), not only for 1. Uniform draws give
  unequal magnitudes, so the error isn't a corner of the hypercube.
- It matters because it hits exactly the panels carrying the headline claim
  (axis-aligned => SSD wins), so it would inflate the result.

BULLET 2: THE SCALE
- The noise level is fixed (nu), but the gradient grows with distance from x*.
  For axis-aligned Q the per-coordinate signal-to-noise ratio is
      |gradL_i| / sigma_i = lambda_i |e_i| / (nu lambda_i) = |e_i| / nu,
  independent of lambda_i. So the scale directly sets the starting SNR.
- U[-10,10] gives RMS |e_i| ≈ 5.8, i.e. a starting SNR of ≈ 58 at nu = 0.1 but
  only ≈ 1.4 at nu = 4. Even at scale 10 the nu = 4 column starts only ~2x above
  the SNR ≈ 1 loss level, ½ nu^2 tr(Q). At scale 1 it starts below that level,
  and every nu = 4 panel is a tie.
- "Noise floor" is loose wording. It's a plateau, not a hard floor. Once the
  SNR drops below ~1 both alpha* shrink toward 0, so progress slows sharply,
  but it doesn't stop.
- At nu = 0 the scale doesn't matter at all: both alpha* formulas and the loss
  are homogeneous in e0, so the curves just rescale by c^2 and the SGD/SSD ratio
  is identical.

"CONCLUSIONS HOLD FOR ANY SCALE ≳ 10": WHAT I CHECKED
I re-ran the grid with x* ~ U[-s, s]. Final SGD/SSD loss ratio after 100 steps
(> 1 means SSD better):

    panel              s=1     3      10     30     100
    ill axis,  nu=0.1  137     750    2740   1170   584
    ill axis,  nu=4    0.90    1.4    9.7    71     543
    well axis, nu=4    0.97    0.91   1.5    2.5    3.7
    ill rot,   nu=0.1  0.98    1.06   0.98   0.78   0.65
    ill rot,   nu=4    0.90    0.92   0.92   0.98   1.02
    well rot,  nu=0.1  0.65    0.86   0.38   0.058  0.006

- The DIRECTION of every comparison is stable for s >= 10. Below that the
  nu = 4 column collapses into ties or flips (well-cond. axis at s = 3).
- The SIZE of the margins is NOT stable. Ill-cond. axis at nu = 4 goes from
  ~10x to ~540x between s = 10 and s = 100. So the magnitudes on slide 9
  (e.g. "SSD 12x") depend on our choice of scale, and only the direction of
  each result is robust. Say this if someone compares our numbers to the
  paper's.
- Ill-cond. rotated drifts from a tie to SGD ~1.5x better at large s. That
  still fits claim 3 (rotation removes SSD's advantage).
- This sweep used one fixed noise seed per run, not the notebook's per-panel
  seeds, so the s = 10 column won't match slide 9 exactly.

---

## 9. Result 1 — sQP

`figures/toy_problem_replication.pdf` — 4 problems × 3 noise levels.

Final $\mathcal{L}(\theta)$ after 100 steps:

| problem | $\nu=0$ | $\nu=0.1$ | $\nu=4$ |
|---|---|---|---|
| well-cond., rotated | SGD $10^{14}\times$ better | SGD 2.6× | SGD 1.4× |
| well-cond., axis-aligned | SGD $10^{10}\times$ | **SSD 4.8×** | **SSD 1.4×** |
| ill-cond., rotated | SGD 1.4× | tie | tie |
| ill-cond., axis-aligned | **SSD 600×** | **SSD 2900×** | **SSD 12×** |

---

## 10. Verdict 1

Reproduced, all four qualitative claims:

1. Noise-free, well-conditioned: SGD wins by orders of magnitude.
2. Noise closes that gap.
3. Random rotation removes SSD's advantage — the methods tie on the ill-conditioned problem.
4. Axis alignment + ill-conditioning: SSD wins by 1–3 orders of magnitude.

Figure is visually near-identical to Fig. 2.

---

## 11. Experiment 2 — M-SVAG

Bias in the moving-average variance estimate, under $\mathbf{E}[m_t]\approx\nabla\mathcal{L}_t$, $\mathbf{E}[v_t]\approx\nabla\mathcal{L}_t^2+\sigma_t^2$:

$$\rho(\beta,t)=\frac{(1-\beta)(1+\beta^{t+1})}{(1+\beta)(1-\beta^{t+1})},
\qquad \mathbf{E}[v_t-m_t^2]\approx(1-\rho)\,\sigma_t^2$$

Corrected estimate and factors:

$$\hat s_t=\frac{v_t-m_t^2}{1-\rho(\beta,t)},
\qquad
\hat\gamma_t^m=\frac{m_t^2}{m_t^2+\rho(\beta,t)\,\hat s_t},
\qquad
\theta_{t+1}=\theta_t-\alpha\,(\hat\gamma_t^m\odot m_t)$$

$\rho$ appears twice: it debiases $\hat s_t$, and it rescales the variance of $m_t$ relative to $g_t$.

Reimplemented in PyTorch as a `torch.optim.Optimizer`; the reference code is TensorFlow.

**Speaker notes — §11**

**What M-SVAG is for.** It fills the empty corner of the grid on slide 3: variance adaptation *without* the sign. It's the control that separates the two parts of Adam. Adam vs. M-SVAG isolates the sign; M-SVAG vs. M-SGD isolates variance adaptation. PyTorch doesn't ship it, so we wrote it.

**Where $\gamma$ comes from (paper's Lemma 1).** We want to step along $p$ but only see a noisy $\hat p$ with $\mathbf{E}[\hat p]=p$ and $\operatorname{var}[\hat p_i]=\sigma_i^2$. Pick per-coordinate factors to minimise $\mathbf{E}\|\gamma\odot\hat p-p\|^2$. Each coordinate is a scalar quadratic in $\gamma_i$:

$$\mathbf{E}\big[(\gamma_i\hat p_i-p_i)^2\big]=\gamma_i^2(p_i^2+\sigma_i^2)-2\gamma_i p_i^2+p_i^2
\quad\Longrightarrow\quad
\gamma_i=\frac{p_i^2}{p_i^2+\sigma_i^2}=\frac{1}{1+\eta_i^2}$$

So each coordinate is shrunk according to how noisy it is relative to its mean. This is the gradient version of Adam's $(1+\eta^2)^{-1/2}$ on the sign (slide 2). With $\hat p=g_t$ you get SVAG. M-SVAG uses $\hat p=m_t$, so it needs the relative variance of $m_t$, not of $g_t$. That's the only reason $\rho$ exists.

**Where $\rho$ comes from.** Unroll the bias-corrected moving average:

$$m_t=\sum_{s=0}^{t}c_s\,g_s,\qquad c_s=\frac{(1-\beta)\,\beta^{t-s}}{1-\beta^{t+1}},\qquad \sum_s c_s=1$$

So $m_t$ is a weighted average of past gradients. Under the slide's assumption (the recent $g_s$ share mean $\nabla\mathcal{L}_t$ and variance $\sigma_t^2$) and independent noise across steps,

$$\operatorname{var}[m_t]=\sigma_t^2\sum_s c_s^2=\rho(\beta,t)\,\sigma_t^2$$

and the slide's closed form is just this geometric sum evaluated. **$1/\rho$ is the effective number of gradients in the average.** For a plain mean of $n$ samples, $\sum c_s^2=1/n$.

| $t$ (paper, 0-indexed) | 0 | 1 | 4 | 9 | 19 | 49 | $\infty$ |
|---|---|---|---|---|---|---|---|
| $\rho(0.9,t)$ | 1 | 0.50 | 0.20 | 0.11 | 0.067 | 0.053 | $\frac{1-\beta}{1+\beta}=0.0526$ |

- At $t=0$, $\rho=1$: one gradient, so there's nothing to average.
- In the long run $1/\rho=19$. If someone says "$\beta=0.9$ averages over 10 steps", that's the $1/(1-\beta)$ window for the mean. For variance reduction the effective count is $(1+\beta)/(1-\beta)=19$.
- $\rho$ is at its limit by ~50 of 6000 steps, so the $t$-dependence only matters at the very start.

**First use of $\rho$: debiasing $v-m^2$.**

$$\mathbf{E}[v_t]\approx\nabla\mathcal{L}^2+\sigma^2,\qquad
\mathbf{E}[m_t^2]=\mathbf{E}[m_t]^2+\operatorname{var}[m_t]\approx\nabla\mathcal{L}^2+\rho\,\sigma^2
\quad\Longrightarrow\quad
\mathbf{E}[v_t-m_t^2]\approx(1-\rho)\,\sigma^2$$

This is Bessel's correction. The $1/n$ sample variance has expectation $(1-\tfrac1n)\sigma^2$ because the mean is estimated from the same data; here $\tfrac1n\to\rho$. Dividing by $1-\rho$ gives $\hat s_t$. In the long run it's only a factor $1/(1-0.0526)\approx1.056$. It matters most early on: $\times2$ at $t=1$.

**Second use of $\rho$: the factor itself.** Put $\hat p=m_t$ into the lemma, with $\mathbf{E}[m_t]\approx\nabla\mathcal{L}$ and $\operatorname{var}[m_t]\approx\rho\sigma^2$:

$$\gamma^m=\frac{\nabla\mathcal{L}^2}{\nabla\mathcal{L}^2+\rho\,\sigma^2}\quad\xrightarrow{\ \nabla\mathcal{L}^2\approx m_t^2,\ \sigma^2\approx\hat s_t\ }\quad\hat\gamma^m_t=\frac{m_t^2}{m_t^2+\rho\,\hat s_t}$$

This $\rho$ is **not** a bias correction. It says that momentum has already averaged away most of the noise, so there's less left to shrink. Combining both uses in the long run:

$$\rho\,\hat s_t=\frac{\rho}{1-\rho}\,(v_t-m_t^2)\;\longrightarrow\;\frac{1-\beta}{2\beta}\,(v_t-m_t^2)\approx0.056\,(v_t-m_t^2)$$

So $\hat\gamma^m\approx 1/(1+\rho\,\eta^2)$, which only halves a coordinate when $\eta^2\approx19$, i.e. when the per-step gradient noise is about $4.4\times$ the gradient itself.

**Likely questions.**
- *How does it relate to Adam?* Drop both $\rho$'s and use $v-m^2$ as the variance: then $\hat\gamma=m^2/(m^2+v-m^2)=m^2/v$, the square of Adam's magnitude factor $|m|/\sqrt v$ (with $\beta_1=\beta_2$). The $\rho$'s are what account for the fact that the update direction is an average.
- *Why the same $\beta$ for $m$ and $v$?* The paper's reason is that it's one averaging window, so one assumption. There's also a practical one: with shared weights, $v_t\ge m_t^2$ always (a weighted mean of squares is at least the square of the weighted mean), so the variance estimate can't go negative. Adam's $\beta_2\neq\beta_1$ loses that guarantee. In our code the `clamp(min=0)` only catches float rounding.
- *Why is $\alpha_{\text{M-SVAG}}=0.3$ bigger than $\alpha_{\text{M-SGD}}=0.1$?* $\hat\gamma\le1$ shrinks every step, so it tolerates a larger global step.

**Where our PyTorch code differs from the paper.** Indexing is fine: the code's `step` is 1-indexed, so `1 - beta**t` is the paper's $1-\beta^{t+1}$, and $\rho$ matches Alg. 2. Two deviations, both measured on a P1 run (M-SVAG, $\alpha=0.3$, 900 steps):

1. **First step.** The paper (§5.4) says $\hat s$ is ill-defined at $t=0$ "since $\rho(\beta,0)=0$". That's a typo: $\rho(\beta,0)=1$, so $1-\rho=0$ and $\hat s_0=0/0$. The paper sets $\hat s_0=0$, which gives $\hat\gamma=1$, a plain SGD first step. Our code divides by $1-\rho+\varepsilon=10^{-8}$, so float32 rounding in $v-m^2$ gets multiplied by $10^8$. Result: the step-1 update is $0.93\times$ the paper's in norm, and $\hat\gamma$ is far from 1 on most coordinates. That's one step out of 6000.
2. **$\varepsilon$ in $\hat\gamma$'s denominator.** The paper uses none; it guards $m=v=0$ explicitly. Ours adds $10^{-8}$. On P1, 50–65% of coordinates have $m_t^2<10^{-8}$, so there $\varepsilon$ dominates and crushes $\hat\gamma$:

| step | mean $\hat\gamma$, paper formula | mean $\hat\gamma$, our code | update norm, code / paper |
|---|---|---|---|
| 1 | 1.00 | 0.47 | 0.93 |
| 10 | 0.59 | 0.30 | 0.98 |
| 200 | 0.28 | 0.15 | 0.99 |
| 900 | 0.33 | 0.18 | 0.99 |

The coordinates where $\varepsilon$ dominates barely move anyway, so the actual update is only ~1% smaller. It's very unlikely to explain the 1–1.5 pp gap on slide 14. Don't quote $\hat\gamma$ statistics from our code as if they were the paper's, though.

With the paper's formula, mean $\hat\gamma\approx0.3$ by coordinate count at batch 64, so variance adaptation really is doing something on P1: most coordinates are noise-dominated.

---

## 12. P1 setup

CNN: conv 5×5×32 → maxpool 3×3/2 → conv 5×5×64 → maxpool 3×3/2 → FC 1024 → FC 10. ReLU, cross-entropy, no regularisation.

Batch 64, 6000 steps, constant $\alpha$, $\beta=0.9$; Adam at defaults.

Step sizes taken from the paper's Appendix A rather than re-tuned:

$$\alpha_{\text{M-SGD}}=10^{-1},\quad
\alpha_{\text{Adam}}=10^{-3},\quad
\alpha_{\text{M-SSD}}=3\cdot10^{-4},\quad
\alpha_{\text{M-SVAG}}=3\cdot10^{-1}$$

---

## 13. Result 2 — Fashion-MNIST

`figures/fashion_mnist_comparison.pdf`. Test accuracy at 6000 steps:

| method | reproduced | paper (approx.) |
|---|---|---|
| Adam | 90.6% | 91.8% |
| M-SSD | 90.0% | 91.5% |
| M-SVAG | 89.8% | 91.0% |
| M-SGD | 89.5% | 90.8% |

Ordering matches: sign-based above non-sign-based; variance adaptation above its base method within each pair.

---

## 14. Caveats

- **1 seed, not 10.** No error bars. The 1.1 pp spread across four methods is not separated from seed noise. Figure title still says 10 seeds.
- Absolute accuracy ~1–1.5 pp below the paper throughout. Uniform offset, so the ordering is unaffected. Likely: PyTorch vs. TensorFlow initialisation, no step-size retuning.
- Training loss not logged — only test accuracy. Claim 1 ("sign dominates") is checked on the weaker of the two panels.
- P3 (CIFAR-100) is the experiment carrying the generalisation claim. Not run. That claim is untested here.

---

## 15. Conclusions

- sQP analysis: reproduces cleanly. Two undocumented details ($x^\ast$, SSD step-size denominator) had to be resolved; neither changes the conclusions.
- P1: ordering reproduces, margins do not separate from noise at 1 seed.
- Untested: the generalisation claim, and the problem-dependence of the sign (P2, P4).

Next: 10 seeds on P1, log training loss, then P2.
