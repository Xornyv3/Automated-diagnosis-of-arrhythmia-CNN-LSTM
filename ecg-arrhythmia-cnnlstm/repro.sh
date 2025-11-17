#!/usr/bin/env bash
set -euo pipefail

# Reproducibility and quieter TF logs
export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
# Disable GPU by default; unset to enable if container has GPUs and TF-GPU is installed
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

DATA_DIR=${DATA_DIR:-data/mitdb}
MODELS_DIR=${MODELS_DIR:-models}
LOGDIR=${LOGDIR:-runs}
FOLDS=${FOLDS:-2}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-3}
SEED=${SEED:-42}

echo "[1/3] Downloading data to ${DATA_DIR} (if needed)"
python -c "from ecgclf.data import download_mitbih; download_mitbih('${DATA_DIR}')"

echo "[2/3] Training ${FOLDS} folds (epochs=${EPOCHS}, batch=${BATCH_SIZE})"
python -m ecgclf.train \
  --data_dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --fold_k "${FOLDS}" \
  --early_stopping \
  --seed "${SEED}" \
  --logdir "${LOGDIR}" \
  --models_dir "${MODELS_DIR}"

echo "[3/3] Evaluating saved fold models"
python -m ecgclf.evaluate \
  --data_dir "${DATA_DIR}" \
  --models_dir "${MODELS_DIR}" \
  --logdir "${LOGDIR}/eval" \
  --fold_k "${FOLDS}" \
  --seed "${SEED}" \
  --fold_avg

echo "Done. TensorBoard logs: ${LOGDIR}"