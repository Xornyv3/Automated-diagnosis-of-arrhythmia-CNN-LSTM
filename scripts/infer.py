import argparse
from pathlib import Path
import csv
import numpy as np

from ecg_arrhythmia.inference import predict_record, CLASS_IDX_TO_NAME


def main():
    p = argparse.ArgumentParser(description="Inference CLI: classify ECG beats for a WFDB record")
    p.add_argument("--record-path", type=str, required=True, help="Path to WFDB record (without extension), e.g., data/mitdb/100")
    p.add_argument("--model-path", type=str, required=True, help="Path to trained Keras model, e.g., models/fold_1.keras")
    p.add_argument("--out", type=str, default="logs/predictions.csv", help="Output CSV path")
    args = p.parse_args()

    res = predict_record(args.record_path, args.model_path)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Write per-beat predictions
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["r_index", "symbol", "true_idx", "pred_idx", "pred_symbol"])  # probs omitted for brevity
        for r, sym, yt, yp in zip(res["r_indices"], res["symbols"], res["y_true"], res["y_pred"]):
            w.writerow([int(r), sym, int(yt), int(yp), CLASS_IDX_TO_NAME.get(int(yp), "?")])

    # Print summary
    y_true = res["y_true"][res["y_true"] >= 0]
    y_pred = res["y_pred"][res["y_true"] >= 0]
    if len(y_true) > 0:
        acc = float(np.mean(y_true == y_pred))
        print(f"Wrote {len(res['y_pred'])} beats to {outp}. Accuracy on labeled beats: {acc:.3f}")
    else:
        print(f"Wrote {len(res['y_pred'])} beats to {outp}.")


if __name__ == "__main__":
    main()
