from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_aggregate(eval_dir: Path) -> Dict:
    agg_path = eval_dir / "aggregate_metrics.json"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics JSON at {agg_path}. Run evaluation with --fold_avg.")
    return json.loads(agg_path.read_text(encoding="utf-8"))


def format_float(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x:.4f}"


def render_report(template: str, agg: Dict, eval_dir: Path) -> str:
    # Basic placeholders
    acc_mean = agg.get("accuracy_mean")
    acc_std = agg.get("accuracy_std")

    template = template.replace("{{ACCURACY_MEAN}}", format_float(acc_mean))
    template = template.replace("{{ACCURACY_STD}}", format_float(acc_std))

    # Per-class means
    def fill_vector(key: str, prefix: str):
        vals: List[float] | None = agg.get(key)
        if not isinstance(vals, list):
            return
        for i, v in enumerate(vals):
            template_key = f"{{{{{prefix}_MEAN_{i}}}}}"
            template_val = format_float(float(v))
            nonlocal_template = locals()
            nonlocal_template
            # Note: We'll rebuild template later; this scope is fine
            nonlocal_rendered[template_key] = template_val

    nonlocal_rendered: Dict[str, str] = {}
    fill_vector("precision_mean", "PREC")
    fill_vector("recall_mean", "REC")
    fill_vector("specificity_mean", "SPEC")
    fill_vector("f1_mean", "F1")

    for k, v in nonlocal_rendered.items():
        template = template.replace(k, v)

    # Confusion matrix path (try common locations)
    candidates = [
        eval_dir / "aggregate_confusion_matrix.png",
        Path("artifacts/figures/aggregate_confusion_matrix_norm.png"),
    ]
    cm_path = next((p for p in candidates if p.exists()), None)
    cm_rel = cm_path.as_posix() if cm_path is not None else ""
    template = template.replace("{{AGG_CM_PATH}}", cm_rel)

    return template


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Markdown report from evaluation artifacts")
    ap.add_argument("--eval_dir", type=str, default="runs/eval", help="Directory with evaluation outputs (aggregate_metrics.json)")
    ap.add_argument("--template", type=str, default="reports/rapport.md", help="Template Markdown file with placeholders")
    ap.add_argument("--out", type=str, default="reports/report.md", help="Output Markdown report path")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    template_path = Path(args.template)
    out_path = Path(args.out)

    agg = load_aggregate(eval_dir)
    template = template_path.read_text(encoding="utf-8")
    rendered = render_report(template, agg, eval_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
