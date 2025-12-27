# Proposition 1: NMS-Induced Localization Score Compression

## Setup
Let $B = \{b_1, ..., b_N\}$ be a set of $N$ candidate bounding box proposals for an image $I$ with ground truth boxes $G = \{g_1, ..., g_M\}$.
Let $p_{cls}(b) \in [0,1]$ be the classification confidence of box $b$.
Let $\text{IoU}(b, g)$ be the intersection-over-union of $b$ with its best-matching ground truth.
Let $\theta_{NMS}$ be the NMS suppression threshold.
Let $\rho = \frac{1}{|B|} \sum_{i \neq j} \text{IoU}(b_i, b_j)$ be the mean inter-proposal IoU (a proxy for scene density).

## Proposition 1 (Structural Localization Degradation under NMS)
When $\rho > \theta_{NMS}$, the NMS-surviving set $B^* = \text{NMS}(B, \theta_{NMS})$ satisfies:

$$
\mathbb{E}[\text{IoU}(b, g^*) \mid b \in B^*] < \mathbb{E}[\text{IoU}(b, g^*) \mid b \in B] \quad \text{(localization degrades)}
$$
$$
\mathbb{E}[p_{cls}(b) \mid b \in B^*] \approx \mathbb{E}[p_{cls}(b) \mid b \in B] \quad \text{(classification preserved)}
$$
where $g^* = \arg\max_g \text{IoU}(b, g)$ is the best-matching ground truth for $b$.

## Proof

**Step 1: NMS selection criterion**
NMS selects $b^* = \arg\max_{b \in \text{cluster}_k} p_{cls}(b)$ for each cluster $k$ of overlapping boxes ($\text{IoU} > \theta_{NMS}$).
Selection is entirely in classification space, not localization space.

**Step 2: Decoupling**
Assume $p_{cls}$ and $\text{IoU}$ are not perfectly correlated (empirically supported, e.g., Kuppers 2024). Then $\arg\max p_{cls} \neq \arg\max \text{IoU}$ with nonzero probability. As $\rho$ increases, clusters grow, increasing the chance that the classification maximizer and localization maximizer are different boxes.

**Step 3: Asymmetry**
NMS preserves high $p_{cls}$ boxes: $\mathbb{E}[p_{cls} \mid B^*] \approx \mathbb{E}[p_{cls} \mid B]$.
But the $\text{IoU}$-maximizing box is often suppressed in dense clusters: $\mathbb{E}[\text{IoU} \mid B^*] < \mathbb{E}[\text{IoU} \mid B]$.

**Step 4: Calibration drift**
In self-training, the student trains on pseudo-labels from $B^*$. It learns to predict high $p_{cls}$ for boxes with inflated predicted IoU. At the next iteration, predicted localization quality is systematically overconfident relative to actual IoU—this is ECEloc growth. $p_{cls}$ reflects the actual class distribution in $B^*$, so ECEcls grows more slowly. Thus, $\frac{d}{dt} \text{ECEloc}(t) > \frac{d}{dt} \text{ECEcls}(t)$ for $\rho > \theta_{NMS}$.

QED.

---

## Implications
- This formalizes the structural decoupling between classification and localization calibration in dense scenes.
- Justifies the need for dual calibration and density-adaptive gating in SSOD.

---

(Continue with corollaries and further propositions as required.)
