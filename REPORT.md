# Short Report: ECG Arrhythmia Classification

## Data

- MIT-BIH Arrhythmia Database (PhysioNet), 360 Hz sampling, MLII lead used where available.
- Classes: Normal (N), LBBB (L), RBBB (R), APB (A), PVC (V).
- Beat-centric segmentation: 1000-sample window centered on R-peak, zero-padded or truncated at boundaries.
- Per-segment z-score normalization.

## Model

- Keras (TensorFlow):
  - 3 × Conv1D + ReLU + MaxPool
  - LSTM(20, recurrent_dropout=0.2)
  - Dense(20) → Dropout(0.2) → Dense(10) → Dropout(0.2)
  - Dense(5) Softmax
- Adam (lr=1e-3), class weights, early stopping on val_loss.

## Evaluation

- 10-fold stratified (by beat labels).
- Metrics: accuracy, per-class sensitivity (recall), specificity, PPV (precision), confusion matrix.
- TensorBoard logs per fold.

## Notes

- Only the 5 classes are used; other annotated beats are ignored.
- Some records use lead V1; we prioritize MLII; records lacking MLII may be skipped.
- Class imbalance handled via class weights.

## Reproducibility

- Fixed random seeds where applicable (NumPy, TF); due to GPU/parallelism, small variations may persist.
