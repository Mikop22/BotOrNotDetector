# Bot Detector Stacking Ensemble

Level-2 stacking pipeline for bot detection using three complementary base learners and a logistic regression meta-learner:

| Model | Signal | Features |
|-------|--------|----------|
| **XGBoost** | Metadata & temporal | 35 hand-crafted features |
| **XLM-RoBERTa** | Linguistic | Fine-tuned on post text |
| **BotRGCN** | Relational | GNN on behavioural kNN graph |
| **Meta-Learner** | Stacked | 8 features (3 probs + interactions + disagreement) |

## Submission Contents

| File | Purpose |
|------|---------|
| `bot_detector.ipynb` | Metadata feature engineering, XGBoost CV, exports `rgcn_features/xgb_features.npz` |
| `bot_detector_xlmr.ipynb` | XLM-R fine-tuning (5-fold), exports `xlmr_cv/` checkpoints and `rgcn_features/xlmr_features.npz` |
| `bot_detector_ensemble.ipynb` | Stacking pipeline: OOF predictions, meta-learner, full-data training, inference on new data, bot ID export |

## Run Order

Run the three notebooks **sequentially** in this exact order:

### Step 1: `bot_detector.ipynb`
- Loads training datasets (`dataset.posts&users.{30-33}.json` + `dataset.bots.{30-33}.txt`)
- Builds 35 metadata/temporal features per user
- Runs 5-fold stratified XGBoost CV
- **Exports:** `rgcn_features/xgb_features.npz` (OOF leaf features + labels + author IDs)

### Step 2: `bot_detector_xlmr.ipynb`
- Preprocesses post text (emoji demojization, URL/mention replacement)
- Fine-tunes `xlm-roberta-base` with 5-fold stratified CV (3-phase training)
- **Exports:** `xlmr_cv/fold{1-5}_phase3/` (model checkpoints) and `rgcn_features/xlmr_features.npz` (OOF [CLS] embeddings)

### Step 3: `bot_detector_ensemble.ipynb`
- Rebuilds XGBoost OOF predictions (with early stopping)
- Loads XLM-R fold checkpoints and produces OOF text probabilities
- Trains BotRGCN on a kNN behavioural graph (with fold-safe scaling and cosine LR schedule)
- Trains a logistic regression meta-learner on 8 stacked features (auto-selects raw vs calibrated variant via nested CV, tunes regularization C)
- Runs full-data training for deployment
- Runs inference on new unseen data (`dataset.posts&users.34.json`, etc.)
- **Exports:** `predicted_bots.txt` (one bot ID per line, competition submission format)

## Environment

Requires Python 3.10+ with:
- `torch`, `torch-geometric`
- `transformers`, `datasets`
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`, `scipy`
- `emoji`, `python-dateutil`

## Data Requirements

Place competition files in the project root:
- **Training:** `dataset.posts&users.{30-33}.json` and `dataset.bots.{30-33}.txt`
- **Test:** `dataset.posts&users.34.json` (or whichever dataset to predict on)

## Output

After running all three notebooks, the final output is:

- **`predicted_bots.txt`** -- predicted bot IDs, one per line (same format as `dataset.bots.*.txt`)
- `new_data_predictions.csv` -- full prediction table with per-model and ensemble probabilities
- `artifacts/ensemble/run_*/` -- reproducibility artifacts (meta-learner, thresholds, calibrators, OOF predictions, configs)

## Leakage-Safety Notes

- OOF protocol: every user's base-learner probability comes from a model that never saw that user during training
- Fold-safe `StandardScaler` for GNN metadata features (fit on train nodes only per fold)
- No outer-val checkpoint selection for BotRGCN (fixed training schedule)
- Robust XLM-R checkpoint resolution (reads `trainer_state.json`, no hardcoded step numbers)
- OOF integrity assertions on aligned `meta_df` columns (coverage, range, fold assignment)
- Calibration variant auto-selected by nested CV; calibrators persisted alongside model when used
