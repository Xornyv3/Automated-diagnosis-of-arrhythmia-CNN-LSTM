from __future__ import annotations
import argparse
import sys
from pathlib import Path


def cmd_prepare(args: argparse.Namespace) -> int:
    """Download MIT-BIH data into the specified directory.

    Optionally prints the list of records downloaded.
    """
    try:
        from .data import download_mitbih
    except Exception as e:
        print(f"[ERROR] Unable to import data tools: {e}")
        return 1

    dest = Path(args.data_dir) if args.data_dir else None
    records = None
    if args.records:
        records = [r.strip() for r in args.records.split(",") if r.strip()]
        if not records:
            records = None

    got = download_mitbih(dest=dest, records=records)
    print(f"Downloaded/verified {len(got)} records to {dest or 'default data dir'}.")
    if args.list:
        print("Records:")
        for r in got:
            print(" ", r)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    try:
        from . import train as train_mod
    except Exception as e:
        print(f"[ERROR] Unable to import training module: {e}")
        return 1
    argv = [
        sys.argv[0],
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--fold_k",
        str(args.fold_k),
        "--seed",
        str(getattr(args, "seed", 42)),
        "--logdir",
        args.logdir,
        "--models_dir",
        args.models_dir,
    ] + (["--early_stopping"] if args.early_stopping else []) + (["--augment"] if args.augment else []) + (["--bandpass"] if args.bandpass else []) + (["--synthetic"] if args.synthetic else []) + (["--synthetic_n", str(args.synthetic_n)] if args.synthetic else [])
    if (not args.synthetic) and args.data_dir:
        argv[1:1] = ["--data_dir", args.data_dir]
    sys.argv = argv
    train_mod.main()
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        from . import evaluate as eval_mod
    except Exception as e:
        print(f"[ERROR] Unable to import evaluation module: {e}")
        return 1
    argv = [
        sys.argv[0],
        "--models_dir",
        args.models_dir,
        "--logdir",
        args.logdir,
        "--fold_k",
        str(args.fold_k),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(getattr(args, "seed", 42)),
    ] + (["--fold_avg"] if args.fold_avg else []) + (["--synthetic"] if args.synthetic else []) + (["--synthetic_n", str(args.synthetic_n)] if args.synthetic else [])
    if (not args.synthetic) and args.data_dir:
        argv[1:1] = ["--data_dir", args.data_dir]
    sys.argv = argv
    eval_mod.main()
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    try:
        from . import predict as pred_mod
    except Exception as e:
        print(f"[ERROR] Unable to import predict module: {e}")
        return 1
    # We simply rebuild argv and delegate to the existing CLI
    argv = [sys.argv[0], "--input", args.input,
            "--model", args.model,
            "--fs", str(args.fs),
            "--pre", str(args.pre), "--post", str(args.post)]
    if args.r_indices:
        argv += ["--r_indices", args.r_indices]
    if args.detect_r:
        argv += ["--detect_r"]
    if args.output_csv:
        argv += ["--out_csv", args.output_csv]
    if args.output_fig:
        argv += ["--out_fig", args.output_fig]
    sys.argv = argv
    pred_mod.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ecgclf",
        description="ECG Arrhythmia classification tools (MIT-BIH) with CNN+LSTM",
        epilog=(
            "Examples:\n\n"
            "  Prepare data (default location):\n"
            "    python -m ecgclf prepare --data_dir data/mitdb\n\n"
            "  Train 10-fold with augmentation and bandpass:\n"
            "    python -m ecgclf train --data_dir data/mitdb --epochs 50 --augment --bandpass\n\n"
            "  Evaluate saved models and produce normalized confusion matrices:\n"
            "    python -m ecgclf evaluate --data_dir data/mitdb --models_dir models --fold_avg\n\n"
            "  Predict on a CSV file with detected R-peaks:\n"
            "    python -m ecgclf predict --input sample.csv --model models/fold_1.keras --detect_r\n"
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "prepare",
        help="Download MIT-BIH records",
        description="Download MIT-BIH Arrhythmia Database records into a local folder.",
    )
    sp.add_argument("--data_dir", type=str, default="data/mitdb", help="Destination directory")
    sp.add_argument("--records", type=str, default="", help="Comma-separated list of record IDs; default uses RECORDS index")
    sp.add_argument("--list", action="store_true", help="Print the list of downloaded/verified records")
    sp.set_defaults(func=cmd_prepare)

    st = sub.add_parser(
        "train",
        help="Train K-fold CNN+LSTM",
        description="Train the CNN+LSTM model with stratified K-fold cross-validation and save models.",
    )
    st.add_argument("--data_dir", type=str, default="data/mitdb", help="Path to mitdb directory (ignored when --synthetic)")
    st.add_argument("--epochs", type=int, default=150)
    st.add_argument("--batch_size", type=int, default=10)
    st.add_argument("--lr", type=float, default=1e-3)
    st.add_argument("--fold_k", type=int, default=10)
    st.add_argument("--seed", type=int, default=42)
    st.add_argument("--early_stopping", action="store_true")
    st.add_argument("--logdir", type=str, default="runs")
    st.add_argument("--models_dir", type=str, default="models")
    st.add_argument("--augment", action="store_true", help="Apply augmentation to training data")
    st.add_argument("--bandpass", action="store_true", help="Apply 0.5-40Hz bandpass before segmentation")
    st.add_argument("--synthetic", action="store_true", help="Use a synthetic offline dataset (no download)")
    st.add_argument("--synthetic_n", type=int, default=200, help="Synthetic samples per class (when --synthetic)")
    st.set_defaults(func=cmd_train)

    se = sub.add_parser(
        "evaluate",
        help="Evaluate saved models",
        description="Evaluate saved fold models, compute metrics (incl. specificity) and save normalized confusion matrices.",
    )
    se.add_argument("--data_dir", type=str, default="data/mitdb", help="Path to mitdb directory (ignored when --synthetic)")
    se.add_argument("--models_dir", type=str, default="models")
    se.add_argument("--logdir", type=str, default="runs/eval")
    se.add_argument("--fold_k", type=int, default=10)
    se.add_argument("--seed", type=int, default=42)
    se.add_argument("--batch_size", type=int, default=128)
    se.add_argument("--fold_avg", action="store_true", help="Also compute mean/std aggregation across folds")
    se.add_argument("--synthetic", action="store_true", help="Use synthetic offline dataset")
    se.add_argument("--synthetic_n", type=int, default=200, help="Synthetic samples per class (when --synthetic)")
    se.set_defaults(func=cmd_evaluate)

    spred = sub.add_parser(
        "predict",
        help="Predict classes for segments from a CSV/.mat signal",
        description="Run inference on a standalone ECG file; optionally detect R-peaks.",
    )
    spred.add_argument("--input", type=str, required=True, help="Path to input CSV or MAT file")
    spred.add_argument("--model", type=str, required=True, help="Path to saved Keras model (.keras)")
    spred.add_argument("--fs", type=int, default=360, help="Sampling frequency of input signal")
    spred.add_argument("--pre", type=int, default=500, help="Samples before R")
    spred.add_argument("--post", type=int, default=500, help="Samples after R")
    spred.add_argument("--r_indices", type=str, default="", help="Optional CSV of R indices")
    spred.add_argument("--detect_r", action="store_true", help="Detect R-peaks automatically")
    spred.add_argument("--output_csv", type=str, default="predictions.csv", help="Where to save predictions CSV")
    spred.add_argument("--output_fig", type=str, default="artifacts/figures/predict_preview.png", help="Where to save preview figure")
    spred.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
