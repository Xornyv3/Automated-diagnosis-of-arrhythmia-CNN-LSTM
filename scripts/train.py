import argparse
from pathlib import Path
import json

from ecg_arrhythmia.data_download import download_mitbih
from ecg_arrhythmia.train_eval import run_kfold_training
from ecg_arrhythmia.config import DATA_DIR


def main():
    p = argparse.ArgumentParser(description="Train ECG arrhythmia classifier with 10-fold stratified CV.")
    p.add_argument("--download", action="store_true", help="Download MIT-BIH (mitdb) to data/mitdb before training")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to mitdb directory")
    p.add_argument("--kfolds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if args.download:
        print("Downloading MIT-BIH to", data_dir)
        recs = download_mitbih(data_dir)
        print(f"Downloaded/verified {len(recs)} records.")

    print("Starting k-fold training…")
    result = run_kfold_training(
        data_dir=data_dir,
        kfolds=args.kfolds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
