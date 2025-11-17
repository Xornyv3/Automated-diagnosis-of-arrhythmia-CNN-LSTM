# Makefile for ecg-arrhythmia-cnnlstm

PY?=python
PIP?=pip
DATA_DIR?=data/mitdb
MODELS_DIR?=models
LOGS_DIR?=runs
ECG?=data/example.csv

.PHONY: setup train eval predict lint test tensorboard typecheck report

setup:
	$(PIP) install -r requirements.txt
	pre-commit install

train:
	$(PY) -m ecgclf train --data_dir $(DATA_DIR) --fold_k 10 --epochs 30 --batch_size 128 --logdir $(LOGS_DIR) --models_dir $(MODELS_DIR)

eval:
	$(PY) -m ecgclf evaluate --data_dir $(DATA_DIR) --models_dir $(MODELS_DIR) --logdir $(LOGS_DIR)/eval --fold_k 10 --seed 42 --fold_avg

predict:
	$(PY) -m ecgclf predict --input $(ECG) --model $(MODELS_DIR)/fold_1.keras --fs 360 --detect_r --output_csv $(LOGS_DIR)/predictions.csv --output_fig $(LOGS_DIR)/predictions.png

lint:
	black src tests
	isort src tests
	flake8 src tests

test:
	pytest -q

tensorboard:
	tensorboard --logdir $(LOGS_DIR)

typecheck:
	mypy src

report:
	$(PY) scripts/make_report.py --eval_dir $(LOGS_DIR)/eval --template reports/rapport.md --out reports/report.md
