# Manuscript Review and Corrected Specification

## 1) Paper Understanding
- Goal: real-time encrypted traffic malicious-flow detection under temporal concept drift and obfuscation.
- Intended contributions: (i) dynamic latent attack representation (DART), (ii) adversarial invariant learning (AGIL), (iii) online transfer updates.
- Deployment model: near-real-time SOC/NIDS flow-level classifier with incremental adaptation.

## 2) Technical Gaps and Risks (Brutal)
- Claimed results appear synthetic/unverifiable in current draft (exact values without protocol artifacts).
- DART equations define static supervised loss but not temporal state transition objective beyond a heuristic update.
- AGIL perturbation space is undefined in feature constraints (which features can be perturbed and physical plausibility bounds).
- Dataset protocol is inconsistent: NSL-KDD is not encrypted traffic and has no true temporal capture chronology.
- CICIDS2017/CICDDoS2019 sourcing is underspecified; direct reproducible download links not fixed.
- Baseline definitions are not reproducible (missing architecture, tuning ranges, code references).
- Latent-similarity metric lacks exact sampling and aggregation protocol.
- Few-shot adaptation protocol conflicts with streaming/online setup (attack-type arrival assumptions unclear).
- No calibration/reliability protocol despite deployment-critical risk.

Reviewer risk: **High** unless revised to conservative, fully reproducible claims.

## 3) Corrected Formal Method (Minimum Sound Version)
- Input: flow-level vector x_t in R^d and label y_t in {0,1}.
- Encoder q_phi(z|x)=N(mu(x),diag(sigma^2(x))).
- Classifier p_theta(y|z)=Bernoulli(sigmoid(g_theta(z))).
- DART loss: L_DART = BCE(y, y_hat) + beta * KL(q_phi(z|x)||N(0,I)).
- AGIL robust objective: L_AGIL = max_{||delta||_inf <= eps} BCE(y, f(x+delta)).
- Full objective: L = L_DART + lambda_adv * L_AGIL.
- Online TL: warm-started mini-batch updates over newest window W_t with replay buffer R.

## 4) Evaluation Protocol (Reproducible)
- Chronological split: 70/15/15 by timestamp/order.
- Report: Accuracy, Precision, Recall, F1, ROC-AUC, ECE.
- Obfuscation stress tests: IDP/IBP/APR/INP at declared strengths.
- Ablations: full, no_dart (beta=0), no_agil (lambda_adv=0), no_online_tl (disable incremental updates).
- All figure outputs as PDF; bar plots grayscale with hatching.

## 5) Honesty Constraints
- Any unavailable dataset must be flagged as unavailable.
- No performance claims until real execution on downloaded/placed datasets.
- Replace "outperforms" language with "evaluated against" until measured.
