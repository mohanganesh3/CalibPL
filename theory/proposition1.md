# Proposition 1 — NMS-induced quality decoupling grows with competition size

## Setting

### Definition (greedy Non-Maximum Suppression)

Fix an IoU threshold $\theta_{\mathrm{NMS}}\in(0,1)$. Given a finite set of candidate boxes with associated class label $c$ and sorting key (score) $S\in[0,1]$, **greedy class-wise NMS** operates as follows:

1. Sort all candidates of class $c$ in **descending** order of $S$.
2. Initialize the kept set $K\leftarrow\emptyset$.
3. Iterate through the sorted candidates; when considering a box $b$, keep it (add to $K$) iff its IoU with every already-kept box is at most $\theta_{\mathrm{NMS}}$. If kept, it **suppresses** (prevents later keeping of) any remaining candidate whose IoU with $b$ exceeds $\theta_{\mathrm{NMS}}$.

This is the standard operator used in modern detectors (possibly preceded by per-class filtering).

Fix a class $c$ and consider a set of $n\ge 1$ candidate detections (proposals) that all survive class-wise filtering and are **mutually suppressing** under NMS, meaning that for the chosen NMS IoU threshold $\theta_{\mathrm{NMS}}$ the overlap graph on these boxes is a clique (equivalently, any one kept box would suppress all others). In this situation, greedy NMS keeps exactly one element of the set.

For each candidate $i\in\{1,\dots,n\}$ define:

- a **classification confidence** (the NMS sorting key) $S_i\in[0,1]$;
- a **localization quality** random variable $Q_i\in[0,1]$ (e.g., IoU with the matched ground-truth box);
- for any fixed IoU threshold $\tau\in(0,1]$, the **localization correctness event**
  $$L_i := \mathbb{1}\{Q_i\ge \tau\}\in\{0,1\}.$$

Assume the pairs $(S_i,Q_i)$ (equivalently $(S_i,L_i)$) are i.i.d. with a continuous distribution for $S$ (ties have probability zero). Let $F_S$ denote the CDF of $S$.

Greedy NMS on this clique keeps the index
$$I := \operatorname{argmax}_{1\le i\le n} S_i,$$
and outputs the kept score $S^* := S_I$ and kept localization indicator $L^* := L_I$.

This “clique” model is the minimal mathematical abstraction of what happens whenever many highly-overlapping boxes compete in the same spatial neighborhood (which is precisely what becomes common as scene density increases).

We treat the **competition size** $n$ as the proxy for scene density: higher density induces larger NMS-connected components and/or larger cliques (or near-cliques) in the overlap graph, increasing the number of mutually suppressing candidates competing for survival.

## Proposition (formal statement)

**Proposition 1 (extreme-score selection amplifies any tail misalignment).**

Define the conditional localization-success function
$$g(s) := \mathbb{P}(L=1\mid S=s),$$
which is well-defined for almost every $s$ under the continuity assumption.

1) (**NMS selects an extreme statistic of $S$**) The NMS-kept score equals the maximum score in the set:
$$S^* \overset{d}= \max_{1\le i\le n} S_i,$$
and therefore
$$\mathbb{P}(S^* \le s) = F_S(s)^n,\qquad s\in[0,1].$$
In particular, $\mathbb{E}[S^*]$ is non-decreasing in $n$ and $S^*\to 1$ in probability as $n\to\infty$ whenever $\mathbb{P}(S>1-\epsilon)>0$ for all $\epsilon>0$.

2) (**Calibration error for any “quality event” grows with $n$ if the model is overconfident in the score tail**) Suppose there exist constants $s_0\in[0,1)$ and $\delta>0$ such that
$$g(s) \le s - \delta\qquad\text{for all } s\in[s_0,1] \text{ (a.e.)}.\tag{A}$$
Then the expected *overconfidence gap* of the NMS-kept box,
$$\Delta_n := \mathbb{E}[S^* - \mathbb{P}(L^*=1\mid S^*)] = \mathbb{E}[S^* - g(S^*)],$$
satisfies the explicit lower bound
$$\Delta_n \ge \delta\,\mathbb{P}(S^*\ge s_0) = \delta\,(1 - F_S(s_0)^n),$$
which is non-decreasing in $n$ and strictly increasing in $n$ whenever $F_S(s_0)\in(0,1)$.

3) (**Consequent density effect**) If scene density increases the typical NMS competition size $n$ (e.g., larger expected clique size in the overlap graph), then the NMS output distribution shifts more mass into the high-score region $[s_0,1]$. Under (A), the expected overconfidence gap for localization success increases with density at least as fast as $1 - F_S(s_0)^n$.

## Proof

### Proof of (1)
In a clique, greedy NMS processes boxes in descending order of the sorting key $S$. Because all boxes mutually overlap above the threshold, the first processed box suppresses all others, and NMS keeps exactly that first box. Under continuity of $S$, the first processed box is the unique maximizer of $S$ among $\{S_i\}_{i=1}^n$, so $S^*=\max_i S_i$.

For i.i.d. $S_i$ with CDF $F_S$, the CDF of the maximum is
$$\mathbb{P}(S^*\le s)=\mathbb{P}(S_1\le s,\dots,S_n\le s)=F_S(s)^n.$$
Monotonicity of $\mathbb{E}[S^*]$ in $n$ follows from first-order stochastic dominance: $F_S(s)^n$ decreases pointwise in $n$, so $S^*$ stochastically increases.

### Proof of (2)
By the law of total expectation and the definition of $g$,
$$\Delta_n = \mathbb{E}[S^* - g(S^*)] = \mathbb{E}[(S^* - g(S^*))\,\mathbb{1}\{S^*\ge s_0\}] + \mathbb{E}[(S^* - g(S^*))\,\mathbb{1}\{S^*< s_0\}].$$
The second term can be dropped to obtain a lower bound:
$$\Delta_n \ge \mathbb{E}[(S^* - g(S^*))\,\mathbb{1}\{S^*\ge s_0\}].$$
Under assumption (A), for $S^*\in[s_0,1]$ we have $S^*-g(S^*)\ge \delta$ almost surely, hence
$$\Delta_n \ge \mathbb{E}[\delta\,\mathbb{1}\{S^*\ge s_0\}] = \delta\,\mathbb{P}(S^*\ge s_0).$$
Using (1),
$$\mathbb{P}(S^*\ge s_0)=1-\mathbb{P}(S^*<s_0)=1-F_S(s_0)^n.$$
This gives the bound.

### Proof of (3)
Part (3) is a direct interpretation of (2): the lower bound depends on $n$ only through $1-F_S(s_0)^n$, which is non-decreasing in $n$. If density increases the typical competition size $n$, then the expected overconfidence gap increases accordingly.

$\square$

## Interpretation (why this is the NMS asymmetry)

Greedy NMS is an **extreme-score selector**: within each high-overlap competition set, it deterministically picks the maximum of $S$. This increases the kept-score distribution with the competition size $n$.

Localization quality, however, is only coupled to $S$ through the conditional function $g(s)=\mathbb{P}(Q\ge\tau\mid S=s)$. If $g(s)$ saturates below $s$ in the high-score tail (which is exactly “the model can be highly confident yet not tightly localized”), then pushing more mass into that tail *necessarily* increases localization overconfidence among NMS-kept boxes.

This is the mathematical core of the asymmetry: NMS acts directly on $S$ but only indirectly on localization. Dense scenes increase $n$, and $n$ increases the strength of extreme-score selection.

## Remarks and extensions

1) **Beyond cliques (general overlap graphs).** In a general overlap graph, greedy NMS produces a maximal independent set with priority ordering by $S$. Each kept box is still the highest-$S$ element among the subset of boxes it suppresses at the time it is selected. Therefore, the same tail-amplification logic applies locally; the clique analysis isolates the cleanest case and yields explicit bounds.

2) **Why sparse scenes behave better.** Sparse scenes correspond to smaller competition sets (smaller $n$). Then $F_S(s_0)^n$ is closer to $F_S(s_0)$ and $\mathbb{P}(S^*\ge s_0)$ is smaller; the bound predicts weaker amplification of tail misalignment.

3) **What must be empirically verified (not assumed).** The only dataset/model-specific condition in Proposition 1 is the tail misalignment assumption (A), i.e., the existence of a high-score region where localization success probability lags behind the score by a margin. Proposition 1 does not assume *why* (A) holds; it states that if it holds, then NMS will amplify it with density.

4) **Connecting to ECEloc vs ECEcls.** To connect this proposition to calibration errors, one instantiates $L$ as a localization-correctness event (IoU above $\tau$) to obtain a lower bound on localization overconfidence after NMS. A parallel analysis can be written for classification correctness, with its own conditional function $g_{\mathrm{cls}}(s)$. If $g_{\mathrm{cls}}(s)$ tracks $s$ more closely in the tail than $g_{\mathrm{loc}}(s)$ does, then the bound grows faster for localization than for classification as $n$ increases.
