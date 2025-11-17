# ECG Arrhythmia Classification – Short Report

## Introduction

This report summarizes the performance of a CNN+LSTM model trained to classify ECG beats into five classes: Normal (N), LBBB (L), RBBB (R), Atrial premature (A), and Ventricular premature (V). Data is segmented into 1000-sample windows centered on R-peaks and z-score normalized. Evaluation uses stratified 10-fold cross-validation.

## Data and Preprocessing

- Dataset: MIT-BIH Arrhythmia (PhysioNet)
- Sampling rate: 360 Hz
- Segmentation: 1000 samples per beat (centered on R)
- Normalization: per-segment z-score

## Model

- Architecture: 3×Conv1D + MaxPool → LSTM(20) → Dense(20) → Dropout(0.2) → Dense(10) → Dropout(0.2) → Softmax(5)
- Optimizer: Adam (lr=1e-3)
- Loss: sparse categorical cross-entropy

## Results (10-fold average)

- Accuracy (mean ± std): {{ACCURACY_MEAN}} ± {{ACCURACY_STD}}

Per-class metrics (means):

| Class | Precision | Recall (Sensitivity) | Specificity | F1 |
|---|---:|---:|---:|---:|
| N | {{PREC_MEAN_0}} | {{REC_MEAN_0}} | {{SPEC_MEAN_0}} | {{F1_MEAN_0}} |
| L | {{PREC_MEAN_1}} | {{REC_MEAN_1}} | {{SPEC_MEAN_1}} | {{F1_MEAN_1}} |
| R | {{PREC_MEAN_2}} | {{REC_MEAN_2}} | {{SPEC_MEAN_2}} | {{F1_MEAN_2}} |
| A | {{PREC_MEAN_3}} | {{REC_MEAN_3}} | {{SPEC_MEAN_3}} | {{F1_MEAN_3}} |
| V | {{PREC_MEAN_4}} | {{REC_MEAN_4}} | {{SPEC_MEAN_4}} | {{F1_MEAN_4}} |

### Confusion Matrix (aggregated)

If available, the aggregated confusion matrix is shown below.

![Aggregate Confusion Matrix]({{AGG_CM_PATH}})

## Discussion

- Strengths: Beat-centric segmentation and class-balanced training improve generalization across folds.
- Limitations: Single-lead input; simple preprocessing; potential class imbalance remains.
- Future Work: Multi-lead inputs, advanced QRS detection, data augmentation, and hyperparameter tuning.
