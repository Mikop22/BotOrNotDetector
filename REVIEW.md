# Bot Detection Model — Research-Level Review

## Overview

This review covers the three-notebook stacking ensemble for bot detection:

| Notebook | Model | Signal |
|----------|-------|--------|
| `bot_detector.ipynb` | XGBoost | Metadata & temporal features |
| `bot_detector_xlmr.ipynb` | XLM-RoBERTa | Linguistic (post text) |
| `bot_detector_ensemble.ipynb` | BotRGCN + Meta-Learner | Relational (GNN) + stacking |

---

## Strengths

### 1. Diverse, Complementary Base Learners
The three base learners capture orthogonal signals — behavioural metadata (XGBoost), linguistic content (XLM-R), and relational structure (BotRGCN). This is the textbook recipe for effective stacking: low error correlation maximises the ensemble's ability to correct individual model mistakes.

### 2. Rigorous OOF (Out-of-Fold) Protocol
Every user's base-learner probability is produced by a model that never saw that user during training. This prevents the well-known "stacking leakage" failure mode where a meta-learner overfits to in-sample base predictions.

### 3. Progressive Fine-Tuning for XLM-R
The 3-phase schedule (head → top-2 layers → full model) with decreasing learning rates prevents catastrophic forgetting of pre-trained representations and stabilises convergence. This is a well-established best practice for large transformer fine-tuning.

### 4. Class-Imbalance Handling Throughout
Weighted cross-entropy loss (XLM-R, BotRGCN), `scale_pos_weight` (XGBoost), and `compute_class_weights` (GNN) consistently address the bot/human imbalance across all stages.

### 5. Fold-Safe Feature Scaling for GNN
The `StandardScaler` for GNN metadata features is correctly fit on train nodes only in each fold, preventing distribution leakage from validation nodes. This attention to detail is above-average for ML competition code.

### 6. Calibration Auto-Selection via Nested CV
The calibration pipeline (Platt scaling and isotonic regression candidates, selected by inner-CV log-loss) prevents calibration-induced leakage and correctly auto-selects the best strategy per base learner. Calibrators are persisted alongside the meta-learner for reproducible deployment.

### 7. Graph Construction is Label-Free
The kNN graph, mention edges, and bio edges are all constructed from label-free signals (hashtag similarity, temporal similarity, @mentions, bio text). No label information leaks into the graph structure.

### 8. OOF Integrity Assertions
The `assert_oof_integrity` function enforces coverage (every user has a prediction), range (probabilities in [0,1]), and alignment (fold assignments consistent) — important sanity checks that catch silent bugs.

### 9. Edge Dropout Regularisation
Stochastic edge dropout (25%) during GNN training acts as a structural regulariser, reducing overfitting to specific graph patterns and improving generalisation.

### 10. Multi-Seed Stability Report
Testing 3 seeds for the meta-level pipeline quantifies how sensitive the final predictions are to random initialisation, calibration fold splits, and threshold selection — a useful robustness diagnostic.

### 11. Thoughtful Feature Engineering
Features like inter-arrival time (IAT) entropy, burstiness coefficient, screen name entropy, Levenshtein distance (username vs display name), and digit density capture well-known bot behavioural signals from the academic literature.

### 12. Robust XLM-R Checkpoint Resolution
The ensemble notebook's `resolve_best_xlmr_checkpoint` reads `trainer_state.json` for the best checkpoint path and falls back to the latest checkpoint directory, avoiding brittle hardcoded step numbers.

---

## Weaknesses

### 1. Activity Vector Lacks Timezone Normalisation
`activity_vector()` counts posts per UTC hour. If the dataset spans multiple timezones, the hourly distribution conflates geographic variation with temporal behaviour. Bot-detection research typically normalises to local time or uses hour-of-day relative to the user's modal activity peak.

### 2. XLM-R Post Concatenation Uses `[SEP]` as Literal Text
```python
text = " [SEP] ".join(t for t in texts if t)
```
XLM-RoBERTa's actual separator token is `</s>`, not `[SEP]` (which is BERT's separator). The string `[SEP]` is tokenised as regular subwords (e.g., `▁[`, `SE`, `P`, `]`), wasting token budget and losing the structural signal that a special separator token provides. While the model can learn to treat the literal string as a separator, using the correct token would be more sample-efficient.

### 3. MAX_LENGTH = 512 May Truncate Prolific Users
Concatenating all of a user's posts into a single sequence with `max_length=512` means prolific users (dozens of posts) will have most of their text truncated. A more robust approach would be to use a sliding window, hierarchical model, or sample representative posts.

### 4. No Early Stopping or Validation-Based Model Selection for GNN
BotRGCN trains for a fixed 200 epochs with cosine annealing but no early stopping or checkpoint selection based on validation loss. While the cosine schedule provides implicit regularisation, the model may still overfit or underfit depending on the dataset. The README explicitly notes this as a design choice, but it sacrifices potential performance.

### 5. Shallow GNN Architecture
Two RGCN layers means the model can only aggregate information from 2-hop neighbours. For sparse social graphs, this may miss important longer-range structural patterns. However, deeper GNNs risk over-smoothing, so this is a trade-off.

### 6. Only 3 Seeds in Stability Report
Three seeds (42, 1337, 2026) provide limited statistical power for assessing stability. A more rigorous analysis would use 10–20 seeds and report confidence intervals.

### 7. No Hyperparameter Search for GNN
The GNN hyperparameters (hidden_dim=128, dropout=0.3, lr=0.01, weight_decay=5e-4, KNN_K=15, epochs=200) are fixed without any search or sensitivity analysis. The pipeline would benefit from at least a small grid search over dropout, hidden_dim, and KNN_K.

### 8. Text Preprocessing Misses Some URL Patterns
The regex `r"https?://\S+"` only matches URLs with explicit `http://` or `https://` prefixes. URLs like `t.co/abc123` or `bit.ly/xyz` without the protocol prefix are not masked, potentially leaking domain-specific information to the text model.

### 9. `[SEP]` Concatenation Loses Temporal Ordering
Posts are joined with `[SEP]` in dictionary iteration order rather than chronological order. The model cannot learn temporal patterns within the text sequence (e.g., increasing repetitiveness over time).

### 10. Feature `min_post_gap_seconds` is Excluded Without Explanation
The minimum inter-post gap is excluded from both the XGBoost and ensemble feature sets. This feature is potentially very discriminative for bot detection (bots often have unnaturally small minimum gaps). If excluded due to high correlation with other features, this should be documented.

---

## Catastrophic Flaws

### FLAW 1: GNN Mention and Bio Edges Are Not Fold-Gated (Transductive Leakage)

**Location:** `bot_detector_ensemble.ipynb`, Cells 3–4

**Description:** The kNN edges (hashtag similarity, temporal similarity) are correctly rebuilt per fold using `knn_edges_train_targets()`, which restricts all target nodes to the training set. However, the explicit **mention edges** and **bio edges** are precomputed once and reused identically in every fold without any fold-gating:

```python
# Cell 3: Precomputed ONCE, reused in every fold
mention_src, mention_tgt, bio_src, bio_tgt = [], [], [], []
# ... (no fold filtering)

# Cell 4: Added to every fold WITHOUT filtering
all_src = ht_src + ts_src + mention_src + bio_src
all_tgt = ht_tgt + ts_tgt + mention_tgt + bio_tgt
```

This means **val→val edges exist** in the GNN graph during each fold's training and evaluation. During message passing, validation nodes exchange features with other validation nodes. Since node features include OOF XLM-R embeddings (which are discriminative for the bot label), class-correlated information propagates between validation nodes through these unfiltered edges.

**Impact:** The GNN's OOF predictions are contaminated by information leaking between validation nodes. This inflates the GNN's OOF F1 score, which in turn inflates the meta-learner's OOF evaluation metrics. The README explicitly claims "Leakage-Safety" for the pipeline, but this claim is violated by the inconsistency between fold-safe kNN edges and non-fold-safe explicit edges.

**Severity:** Moderate-to-High. The degree of leakage depends on the density of mention/bio edges between validation nodes. In typical social media datasets, mention graphs can be dense, making this a non-trivial source of information leakage.

**Fix applied:** Added fold-gating to filter mention and bio edges so that targets are restricted to training nodes in each fold, consistent with the kNN edge handling.

---

### FLAW 2: Hardcoded Checkpoint Path in XLM-R OOF Export

**Location:** `bot_detector_xlmr.ipynb`, Cell 8

**Description:** The OOF embedding export cell hardcodes a specific checkpoint step number:

```python
with open(f"xlmr_cv/fold{fold}_phase3/checkpoint-135/trainer_state.json") as f:
    best_ckpt = _json.load(f)["best_model_checkpoint"]
```

The step number `135` is specific to the exact dataset size and batch configuration used during the original training run. If the dataset size, batch size, or gradient accumulation steps change, the checkpoints will be saved at different step numbers, and this code will raise a `FileNotFoundError`.

**Impact:** The notebook will crash during the OOF export step if retrained with a different configuration, preventing the ensemble pipeline from running. This is particularly insidious because the XGBoost and XLM-R training cells may complete successfully, wasting hours of GPU time before failing.

**Severity:** High. The ensemble notebook already implements the correct dynamic resolution (`resolve_best_xlmr_checkpoint`), but the XLM-R notebook's own export cell does not use it.

**Fix applied:** Replaced the hardcoded path with dynamic checkpoint resolution logic consistent with the ensemble notebook's `resolve_best_xlmr_checkpoint` function.

---

### FLAW 3: Meta-Learner Hyperparameter Selection Leaks Into OOF Evaluation

**Location:** `bot_detector_ensemble.ipynb`, Cell 6

**Description:** The `eval_meta_stack` function uses the same CV splits for both hyperparameter selection (`GridSearchCV`) and OOF evaluation (`cross_val_predict`):

```python
def eval_meta_stack(X_meta, y_true, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(..., cv=cv, ...)
    grid.fit(X_meta, y_true)       # C selected using ALL data
    best_model = grid.best_estimator_
    probs = cross_val_predict(best_model, X_meta, y_true, cv=cv, ...)
```

`GridSearchCV.fit()` selects the best `C` by evaluating over all folds of the full dataset. This `C` is then used in `cross_val_predict` which re-fits the model on each fold's training data. While `cross_val_predict` correctly produces OOF predictions for the probabilities, the regularisation strength `C` was optimised on data that includes each fold's validation set.

**Impact:** The meta-learner's `C` is tuned with information from validation data, creating a subtle form of hyperparameter leakage. For logistic regression with only 4 candidate values, the practical impact is small, but it violates the strict OOF protocol that the rest of the pipeline carefully maintains. This can lead to mildly over-optimistic reported F1 scores.

**Severity:** Low-to-Moderate. The leakage is bounded by the small hyperparameter search space and the stability of logistic regression. However, it is conceptually inconsistent with the pipeline's stated leakage-safety guarantees.

**Fix applied:** Refactored `eval_meta_stack` to use proper nested cross-validation: the outer loop produces OOF predictions while the inner loop selects `C` using only training-fold data.

---

## Summary

| Category | Count | Key Items |
|----------|-------|-----------|
| **Strengths** | 12 | Diverse ensemble, rigorous OOF protocol, progressive fine-tuning, calibration auto-selection |
| **Weaknesses** | 10 | No timezone normalisation, shallow GNN, truncation at 512 tokens, limited seed sweep |
| **Catastrophic Flaws** | 3 | GNN val→val edge leakage, hardcoded checkpoint path, meta-learner C selection leakage |

The pipeline demonstrates strong ML engineering practices overall — the OOF protocol, calibration handling, and integrity assertions are above the standard for research code. The primary concerns are the GNN fold-gating inconsistency (which undermines the claimed leakage safety), the brittle hardcoded checkpoint path (which blocks reproducibility), and the meta-learner hyperparameter leakage (which is conceptually inconsistent with the pipeline's stated guarantees).
