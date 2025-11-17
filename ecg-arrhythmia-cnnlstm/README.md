# ecg-arrhythmia-cnnlstm

Minimal repository scaffolding for ECG arrhythmia classification (MIT-BIH) using a CNN+LSTM model in TensorFlow.

## Structure

- `src/ecgclf/`: package with data, preprocessing, model, and CLIs (train/evaluate/predict)
- `notebooks/01_demo.ipynb`: quick-start demo
- `tests/`: pytest tests
- `pyproject.toml`: packaging and tool configs (black, isort, flake8)
- `requirements.txt`: dependencies
- `.pre-commit-config.yaml`: formatting/lint hooks
- `Makefile`: handy targets
- `Dockerfile`: container build

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pre-commit install
```

Offline quick run (no downloads):

```powershell
# 2 folds × 3 epochs on a synthetic dataset
python -m ecgclf train --synthetic --synthetic_n 200 --epochs 3 --batch_size 64 --fold_k 2 --early_stopping --logdir runs --models_dir models
python -m ecgclf evaluate --synthetic --synthetic_n 200 --fold_k 2 --batch_size 128 --fold_avg --logdir runs/eval --models_dir models
```

Train (example):

```powershell
python -m ecgclf train --data_dir .\data\mitdb --fold_k 10 --epochs 30 --batch_size 128
```

TensorBoard:

```powershell
tensorboard --logdir .\runs
```

## Contexte & objectifs

Ce projet propose un pipeline complet pour la classification d’arythmies cardiaques à partir d’ECG (base MIT-BIH Arrhythmia, PhysioNet). L’objectif est de segmenter les battements autour des pics R, de normaliser chaque segment (z-score), puis d’entraîner un modèle CNN+LSTM afin de classer les battements parmi 5 classes: Normal, LBBB, RBBB, APB, PVC.

## Schéma du modèle (Keras)

- Entrée: (1000, 1)
- 3 × [Conv1D (kernels ~21/11/5, ReLU, use_bias=False) → MaxPool1D(stride=2)]
- LSTM(20, recurrent_dropout=0.2, return_sequences=False)
- Dense(20, ReLU) → Dropout(0.2)
- Dense(10, ReLU) → Dropout(0.2)
- Dense(5, Softmax)
- Optimiseur: Adam(lr=1e-3), perte: sparse_categorical_crossentropy, métrique: accuracy

Une implémentation PyTorch équivalente est disponible via `build_model(..., backend="torch")` (non compilée automatiquement; entraînement PyTorch à gérer côté utilisateur).

## Prérequis

- Windows, macOS, ou Linux
- Python 3.10/3.11 recommandé pour TensorFlow (les roues TF pour Python 3.13 peuvent être indisponibles)
- Espace disque pour `mitdb` (~ tens of MB)

## Installation

One-liner via Make:

```powershell
make setup
```

Cela installe les dépendances (`requirements.txt`) et les hooks `pre-commit` (black, isort, flake8).

Astuce (Windows sans `make`):

```powershell
pip install -r requirements.txt
pre-commit install
```

## Préparation des données

Les scripts attendent les fichiers MIT-BIH dans `data/mitdb`. Vous pouvez télécharger via la librairie `wfdb` (ex: `ecgclf.data.download_mitbih`) ou déposer manuellement les fichiers `.dat/.hea/.atr`.

Exemple Python (optionnel):

```powershell
python -c "from ecgclf.data import download_mitbih; download_mitbih('data/mitdb')"
```

## Entraînement

Entraînement 10-fold stratifié (sauvegarde des modèles par fold et logs TensorBoard):

```powershell
make train
```

Paramètres par défaut dans `Makefile`. Pour personnaliser:

```powershell
python -m ecgclf.train --data_dir .\data\mitdb --epochs 150 --batch_size 10 --lr 1e-3 --fold_k 10 --early_stopping --seed 42 --logdir runs --models_dir models
```

## Évaluation

Évalue les modèles sauvegardés (par fold) et génère:

- accuracy globale,
- sensibilité (recall) par classe,
- spécificité par classe,
- précision/PPV par classe,
- F1 par classe,
- matrice de confusion (PNG),
- classification report (CSV & Markdown),
- agrégats (moyenne/écart-type) via `--fold_avg`.

```powershell
make eval
```

### Rapport Markdown

Après l'évaluation avec agrégats (`make eval` inclut `--fold_avg`), générez un rapport Markdown prêt à partager:

```powershell
make report
```

Cela produit `reports/report.md` à partir du template `reports/rapport.md` et des artefacts dans `logs/eval`.

## Inférence

Prédire des classes sur un ECG mono-lead (CSV ou .mat). Si vous n’avez pas les indices R, un détecteur simple peut être activé via `--detect_r`.

```powershell
# Exemple avec détection R côté CLI (remplacez le chemin du fichier)
make predict ECG=path\to\ecg.csv

# Ou directement via la méta-commande
python -m ecgclf predict --input .\path\to\ecg.mat --model .\models\fold_1.keras --fs 360 --detect_r --output_csv .\runs\predict\preds.csv --output_fig .\runs\predict\preds.png
```

## Notebook démo

`notebooks/01_demo.ipynb` montre un mini-flux: visualisation de segments, entraînement rapide (quelques époques), courbes loss/accuracy et matrice de confusion.

## Liens utiles

- [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [WFDB (Python)](https://wfdb.readthedocs.io/)

## Avertissements

Projet à visée académique et expérimentale uniquement. Ce code et les modèles entraînés ne constituent pas un dispositif médical et ne doivent pas être utilisés pour un diagnostic clinique.

## Docker (optionnel)

Une image Docker basée sur Python 3.10 est fournie pour exécuter le pipeline.

Caractéristiques:

- `TF_CPP_MIN_LOG_LEVEL=2` (logs TensorFlow réduits)
- `CUDA_VISIBLE_DEVICES=""` (GPU désactivé par défaut; décommentez ou changez pour utiliser un GPU)
- Volume de travail: `/workspace`

Construction de l'image:

```powershell
docker build -t ecgclf:latest .
```

Exécution avec montage du projet courant dans `/workspace` et lancement d'un shell:

```powershell
docker run --rm -it -v ${PWD}:/workspace -w /workspace ecgclf:latest bash
```

Pipeline de reproduction (data → train → eval) avec seed fixé via `repro.sh`:

```powershell
docker run --rm -it -v ${PWD}:/workspace -w /workspace ecgclf:latest bash -lc "./repro.sh"
```

Paramètres (variables d'environnement) par défaut dans `repro.sh`:

- `DATA_DIR=data/mitdb`, `MODELS_DIR=models`, `LOGDIR=runs`, `FOLDS=2`, `EPOCHS=5`, `BATCH_SIZE=64`, `LR=1e-3`, `SEED=42`.

Exemple avec 10 folds et plus d'époques:

```powershell
docker run --rm -it -v ${PWD}:/workspace -w /workspace -e FOLDS=10 -e EPOCHS=30 ecgclf:latest bash -lc "./repro.sh"
```
