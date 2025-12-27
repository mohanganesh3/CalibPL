# Calibrator optimality + gating geometry (what CalibPL is optimizing)

This note is *not* a sales pitch.
It is an explicit statement of what the current CalibPL gate is (and is not) optimal for, under clean assumptions.

The core point: **given only scalar “classification” and “localization” scores, the optimal pseudo-label acceptance region is generally *not* a rectangle**. CalibPL’s rectangular (AND) gate is a conservative approximation that trades recall for guaranteed joint reliability; this is *good* in dense regimes (noise amplification) and can be *harmful* in sparse regimes (starvation) unless the thresholds adapt.

---

## 1) Setting: pseudo-label acceptance as a decision rule

Consider a candidate detection/proposal producing two scalar scores

- $S_{\mathrm{cls}} \in [0,1]$ (classification score used for ranking / NMS sorting)
- $S_{\mathrm{loc}} \in [0,1]$ (localization score; e.g., predicted IoU proxy)

Define two correctness events for the pseudo label:

- $C := \mathbb{1}\{\text{predicted class is correct}\}$
- $L := \mathbb{1}\{\mathrm{IoU}(\hat b, b_{gt}) \ge \tau\}$ (e.g., $\tau=0.5$)

A pseudo label is *fully correct* when $Y := C \wedge L = 1$.

A pseudo-label filter is a measurable acceptance function

$$A(S_{\mathrm{cls}}, S_{\mathrm{loc}}) \in \{0,1\}.$$

The ideal (information-theoretic) quantity to rank by is the conditional joint correctness

$$p_{\mathrm{joint}}(s_c, s_l) := \mathbb{P}(Y=1 \mid S_{\mathrm{cls}}=s_c,\ S_{\mathrm{loc}}=s_l).$$

If we can compute $p_{\mathrm{joint}}$ exactly, then standard Bayes arguments imply:

- **Budget-constrained optimality:** among all rules that accept exactly $k$ candidates, the rule that accepts the top-$k$ values of $p_{\mathrm{joint}}$ maximizes the expected number of correct pseudo labels.
- **Precision-constrained optimality:** among all rules whose accepted set has conditional correctness at least $\pi$ (i.e., $\mathbb{P}(Y=1 \mid A=1)\ge \pi$), the rule that accepts $\{p_{\mathrm{joint}}\ge \pi\}$ is “maximally permissive” (it accepts every candidate whose *true* joint correctness is at least $\pi$).

These statements are deterministic once $p_{\mathrm{joint}}$ is known.

In practice we do **not** know $p_{\mathrm{joint}}$; CalibPL therefore estimates *marginal* conditional probabilities and composes them.

---

## 2) What the 1D isotonic calibrators estimate

CalibPL fits two 1D calibrators (per iteration $t$):

$$g^{\mathrm{cls}}(s) \approx \mathbb{P}(C=1 \mid S_{\mathrm{cls}}=s),\qquad g^{\mathrm{loc}}(s) \approx \mathbb{P}(L=1 \mid S_{\mathrm{loc}}=s).$$

When the target conditional function is monotone (a standard “score monotonicity” assumption), isotonic regression is a consistent nonparametric estimator of that conditional probability.

**Important limitation:** even with perfect estimation of both $g^{\mathrm{cls}}$ and $g^{\mathrm{loc}}$, we have *not* identified $p_{\mathrm{joint}}(s_c,s_l)$.

---

## 3) Independence factorization and its consequences

A common approximation is conditional independence of the correctness events given their scores:

$$\mathbb{P}(Y=1 \mid s_c,s_l) = \mathbb{P}(C=1, L=1 \mid s_c,s_l)
\ \approx\ \mathbb{P}(C=1 \mid s_c)\,\mathbb{P}(L=1 \mid s_l)
\ =\ g^{\mathrm{cls}}(s_c)\,g^{\mathrm{loc}}(s_l).$$

Under this approximation,

$$p_{\mathrm{joint}}(s_c,s_l) \approx p_{\mathrm{prod}}(s_c,s_l) := g^{\mathrm{cls}}(s_c)\,g^{\mathrm{loc}}(s_l).$$

So the *product score* $p_{\mathrm{prod}}$ becomes the natural scalar to rank by.

### Consequence A (optimal acceptance region is hyperbolic)

If we target a minimum joint reliability $\pi$, the “accept if joint probability exceeds $\pi$” rule becomes

$$A_{\pi}^{\star}(s_c,s_l) = \mathbb{1}\{g^{\mathrm{cls}}(s_c)\,g^{\mathrm{loc}}(s_l) \ge \pi\}.$$

This acceptance region is generally a **hyperbola** in the $(g^{\mathrm{cls}}, g^{\mathrm{loc}})$ plane, not a rectangle.

---

## 4) The CalibPL AND gate is a conservative rectangular approximation

Ignoring CGJS for a moment, CalibPL’s gate is (in calibrated space)

$$A_{\mathrm{AND}}(s_c,s_l) = \mathbb{1}\{g^{\mathrm{cls}}(s_c)\ge r_{\mathrm{cls}}\ \wedge\ g^{\mathrm{loc}}(s_l)\ge r_{\mathrm{loc}}\}.$$

### Proposition 2 (guaranteed joint correctness lower bound under factorization)

Assume the factorization approximation holds exactly for the moment.
Then for any $(s_c,s_l)$,

$$A_{\mathrm{AND}}(s_c,s_l)=1\ \Rightarrow\ p_{\mathrm{prod}}(s_c,s_l) \ge r_{\mathrm{cls}}\,r_{\mathrm{loc}}.$$

**Proof:** If $g^{\mathrm{cls}}(s_c)\ge r_{\mathrm{cls}}$ and $g^{\mathrm{loc}}(s_l)\ge r_{\mathrm{loc}}$, multiply both inequalities. $\square$

So the AND gate is a *sufficient* condition for achieving joint reliability $\pi=r_{\mathrm{cls}}r_{\mathrm{loc}}$.

### Corollary (why AND can starve)

Because the rectangle $\{g^{\mathrm{cls}}\ge r_{\mathrm{cls}},\ g^{\mathrm{loc}}\ge r_{\mathrm{loc}}\}$ is a strict subset of the hyperbola region $\{g^{\mathrm{cls}}g^{\mathrm{loc}}\ge r_{\mathrm{cls}}r_{\mathrm{loc}}\}$, CalibPL **rejects** many candidates that would be accepted by the product rule at the *same* target joint reliability.

This is exactly the *precision–recall trade*:
- AND gate: higher guaranteed joint reliability per accepted label, lower coverage.
- product gate: more coverage at the same joint-reliability target.

In sparse regimes (COCO 1% diagnostics), this can manifest as pseudo-label starvation if $r_{\mathrm{loc}}$ is chosen too aggressively.

---

## 5) Adding CGJS: a third axis and the “triple AND” geometry

With CGJS stability $\rho(b)\in[0,1]$ and threshold $\beta$, the full CalibPL acceptance is

$$A=\mathbb{1}\{g^{\mathrm{cls}}\ge r_{\mathrm{cls}}\ \wedge\ g^{\mathrm{loc}}\ge r_{\mathrm{loc}}\ \wedge\ \rho\ge\beta\}.$$

Even if we interpret $\rho$ as a probability-like correctness proxy, the same geometric point remains: this is a **rectangular box** in $(g^{\mathrm{cls}}, g^{\mathrm{loc}}, \rho)$.

A continuous alternative (suggested by the limitations section) is to compute a *single scalar* and threshold it, e.g.

$$p_{\mathrm{scalar}} := g^{\mathrm{cls}}\cdot g^{\mathrm{loc}}\cdot \rho,\qquad A=\mathbb{1}\{p_{\mathrm{scalar}}\ge \tau\}.$$

This is closer to the “accept high joint correctness” principle, and avoids discontinuities of the AND gate.

---

## 6) What *is* optimal, and under what assumptions?

**Claim we can defend:**
- If $g^{\mathrm{cls}}$ and $g^{\mathrm{loc}}$ are well-calibrated estimates of their respective conditional probabilities, and if joint correctness approximately factorizes, then ranking candidates by the product $g^{\mathrm{cls}}g^{\mathrm{loc}}$ is the natural approximation to Bayes-optimal ranking by $p_{\mathrm{joint}}$.

**Claim we should *not* overstate without evidence:**
- The rectangular AND rule is optimal. It generally is not; it is a conservative inner approximation.

**Empirical consequence (testable):**
- If a learned joint calibrator (2D) does *not* improve pseudo-label selection / downstream AP over the product rule, then the factorization approximation is empirically adequate for our setting even when the raw errors are correlated.

This is why the “joint 2D calibrator” experiment is not optional if we want to fully justify the independence assumption.

---

## 7) Why density-adaptive localization gating is principled

Proposition 1 shows that NMS tail amplification (hence localization overconfidence of kept boxes) scales with competition size $n$ / density.

- Dense regime: tail amplification strong → conservative gate (higher $r_{\mathrm{loc}}$, higher $\beta$) can be necessary to prevent compounding localization noise.
- Sparse regime: tail amplification weak → same conservative gate can reduce pseudo-label coverage below the regime needed to beat supervised training.

Thus, making $r_{\mathrm{loc}}$ (and/or $\beta$) a function of a density proxy is the correct response to the fact that the *selection-bias strength* is density-dependent.

---

## 8) Immediate manuscript integration points

- In the method section, explicitly state that the AND gate is a conservative approximation to a product (joint-probability) rule, and that density-adaptive localization gating exists to avoid starvation in sparse regimes.
- In limitations, be explicit: a learned joint calibrator could recover some of the rejected hyperbola mass and improve recall at fixed precision; we report it as future work unless implemented.
