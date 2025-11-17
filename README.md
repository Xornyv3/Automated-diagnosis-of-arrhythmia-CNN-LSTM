# Automated diagnosis of arrhythmia using CNN and LSTM

Automatic diagnosis of arrhythmia using a hybrid Convolutional Neural Network (CNN) + Long Short-Term Memory (LSTM) model that supports variable-length heartbeat segments.

> Short: Automated diagnosis of arrhythmia

**Badges:** add CI / coverage / license badges here

---

## Table of contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Data & Models](#data--models)
- [Development](#development)
- [Contributing](#contributing)
- [License & Citation](#license--citation)

---

## Overview

This repository implements an end-to-end pipeline for ECG arrhythmia classification. The core approach combines CNNs (for spatial/shape features) with LSTMs (for temporal context) and is designed to handle variable-length heartbeat segments.

Primary components:

- `src/` — library code (data loaders, preprocessing, model architecture, training, metrics)
- `scripts/` — CLI entrypoints (`train.py`, `infer.py`, utility scripts)
- `notebooks/` — demo and reproducibility notebooks
- `data/`, `models/`, `runs/`, `logs/` — runtime artifacts (ignored from git)

## Quickstart

Recommended: use a virtual environment and install dependencies.

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

If you have a CPU-only environment, consider installing `tensorflow-cpu` instead of `tensorflow`.

## Usage

Train (example):

```powershell
python scripts/train.py --kfolds 10 --epochs 30 --batch-size 128 --patience 5 --lr 1e-3
```

This will run training, save checkpoints to `runs/` (or `models/` depending on config), and produce TensorBoard logs under `logs/tensorboard/`.

View logs:

```powershell
tensorboard --logdir .\\logs\\tensorboard
```

Inference (example):

```powershell
python scripts/infer.py --record-path data/mitdb/100 --model-path runs/mitdb/best_model.keras --out logs/pred_100.csv
```

Check `notebooks/01_demo.ipynb` for an end-to-end demonstration.

## Data & Models

- Data: Place raw datasets under `data/` (e.g., `data/mitdb/`). The repository's `.gitignore` is configured to avoid committing dataset files. Do not push raw patient data to GitHub.
- Models & artifacts: Trained weights, checkpoints, and evaluation artifacts live in `models/`, `runs/`, or `artifacts/`. Use external storage (S3, Google Drive, or an artifact registry) for sharing large files.

## Development

- Run tests:

```powershell
pytest -q
```

- Lint/format: follow `pyproject.toml` / pre-commit hooks if present.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository and create a feature branch
2. Add tests for new functionality
3. Open a pull request with a clear description and reproducible steps

For large data/models, include download helpers (scripts/download_*) and a short README describing required data structure.

## License & Citation

- License: add a `LICENSE` file (MIT recommended for permissive use) and specify terms here.
- Data source: PhysioNet MIT-BIH Arrhythmia Database — cite per PhysioNet guidelines when publishing.

---

If you want, I can also:

- add CI badges and a short demo GIF to the top of this `README.md`
- commit this change for you
- create a `CONTRIBUTING.md` or `CODE_OF_CONDUCT.md`

Files touched: `README.md`

## Results (aggregate)

Below are the aggregate evaluation metrics computed across 10 stratified folds. For full interactive report and downloadable artifacts see `ecg-arrhythmia-cnnlstm/web-static/index.html`.

| Class  | Precision (PPV) | Recall (Sensitivity) | Specificity | F1    | Support |
|--------|-----------------:|---------------------:|------------:|:-----:|-------:|
| Normal | 0.9861           | 0.8231               | 0.9669      | 0.8942| 7505   |
| LBBB   | 0.9541           | 0.9863               | 0.9957      | 0.9695| 807    |
| RBBB   | 0.9212           | 0.9596               | 0.9931      | 0.9385| 725    |
| APB    | 0.4127           | 0.7663               | 0.9646      | 0.5186| 254    |
| PVC    | 0.4643           | 0.9194               | 0.8975      | 0.6053| 713    |

Aggregate accuracy (mean over folds): **0.8516**

### Confusion matrix & figures

The project exports visualizations (confusion matrix, example beats, training curves) to the `ecg-arrhythmia-cnnlstm/web-static/` folder. If the PNGs are present they will appear in the static report; if you want them embedded directly in this `README.md`, add the image files to the repo (suggested paths):

- `ecg-arrhythmia-cnnlstm/web-static/aggregate_confusion_matrix_norm.png` — normalized confusion matrix
- `ecg-arrhythmia-cnnlstm/web-static/examples.png` — representative beats per class
- `ecg-arrhythmia-cnnlstm/web-static/training_curves.png` — training/validation loss & accuracy plots

Example: to embed the confusion matrix directly, add the file above and then include:

```markdown
![Normalized confusion matrix](ecg-arrhythmia-cnnlstm/web-static/aggregate_confusion_matrix_norm.png)
```

If you want, I can:

- search the workspace for the figure PNGs and embed them automatically into the `README.md` (I already checked `ecg-arrhythmia-cnnlstm/web-static/` and found the HTML and JSON/CSV report; the PNG images were not present), or
- generate the confusion matrix and example plots from available metrics & sample data and commit the images here (I can run a small script to create PNGs if you want me to and if you confirm it's OK to run Python here).

