"""
02_analyze_hallucination.py

Computes IDK Rate and Hallucination Rate per model from eval.py's output,
and prints a comparison table across all 5 models.

Run this AFTER eval.py has been run for all models.

Usage:
    python scripts/02_analyze_hallucination.py \
        --results_dir outputs/evaluations \
        --pred_dir outputs/predictions \
        --csv data/test_half_release.csv

Inputs (per model):
    outputs/evaluations/results_{model_name}.json  — per-question eval output from eval.py
    outputs/predictions/{model_name}.jsonl          — raw model predictions

Outputs:
    Printed summary table + outputs/final/hallucination_analysis.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tabulate import tabulate


# Keywords indicating a model answered "I don't know"
IDK_PATTERNS = [
    r"i don.t know",
    r"i do not know",
    r"cannot determine",
    r"can.t determine",
    r"not (visible|clear|shown|seen|apparent|legible)",
    r"unable to (tell|see|determine|read|identify)",
    r"can.t tell",
    r"no (way to|information)",
]
IDK_REGEX = re.compile("|".join(IDK_PATTERNS), re.IGNORECASE)


def is_idk_response(text: str) -> bool:
    """Returns True if the model's prediction expresses uncertainty/inability."""
    return bool(IDK_REGEX.search(text))


def is_unanswerable_gt(row) -> bool:
    """Returns True if any ground-truth answer for this question is 'I don't know'."""
    for col in ["answer0", "answer1", "answer2", "answer3"]:
        val = getattr(row, col, None)
        if pd.notna(val) and isinstance(val, str) and is_idk_response(val):
            return True
    return False


MODEL_NAMES = [
    "videollama3",
    "internvl2_5",
    "internvl3_5",
    "llava_onevision",
    "qwen2_5_vl",
    "videochat_r1",
    "qwen3_vl",
    "gemma4",
    "glm4_1v",
]

# Paper-reported overall accuracy for reference
PAPER_ACCURACY = {
    "videollama3":    49.2,
    "internvl2_5":    53.5,
    "internvl3_5":    float("nan"),
    "llava_onevision": 54.5,
    "qwen2_5_vl":     45.5,
    "videochat_r1":   float("nan"),
    "qwen3_vl":       float("nan"),
    "gemma4":         float("nan"),
    "glm4_1v":        float("nan"),
}


def load_predictions(pred_path: Path) -> dict:
    """Load JSONL predictions into a dict: question_id -> pred text."""
    preds = {}
    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["question_id"]] = obj.get("pred", "")
    return preds


def load_results(results_path: Path) -> dict:
    """Load eval.py results JSON: question_id -> {'pred': 'yes/no', 'score': float}
    eval.py stores each entry as [response_dict, qa_set]; unwrap to response_dict only."""
    with open(results_path, "r") as f:
        data = json.load(f)
    return {k: v[0] if isinstance(v, list) else v for k, v in data.items()}


def analyze_model(model_name: str, df: pd.DataFrame, results_dir: Path, pred_dir: Path) -> dict:
    """Compute all metrics for a single model."""
    results_file = results_dir / f"result_{model_name}.json"
    pred_file = pred_dir / f"{model_name}.jsonl"

    if not pred_file.exists():
        print(f"  [SKIP] No predictions file for {model_name}: {pred_file}")
        return None
    if not results_file.exists():
        print(f"  [SKIP] No results file for {model_name}: {results_file}")
        return None

    preds = load_predictions(pred_file)
    results = load_results(results_file)

    # Build per-question records
    records = []
    for row in df.itertuples(index=False):
        qid = row.question_id
        gt_unanswerable = is_unanswerable_gt(row)
        pred_text = preds.get(qid, "")
        model_said_idk = is_idk_response(pred_text)

        # eval.py result for this question
        eval_result = results.get(qid, {})
        gpt_correct = str(eval_result.get("pred", "no")).lower() == "yes"

        records.append({
            "question_id": qid,
            "type": row.type,
            "gt_unanswerable": gt_unanswerable,
            "model_said_idk": model_said_idk,
            "gpt_correct": gpt_correct,
        })

    result_df = pd.DataFrame(records)
    unanswerable = result_df[result_df["gt_unanswerable"]]
    answerable = result_df[~result_df["gt_unanswerable"]]

    total = len(result_df)
    n_unanswerable = len(unanswerable)
    n_answerable = len(answerable)

    # Overall GPT score (accuracy)
    gpt_score = result_df["gpt_correct"].mean() * 100

    # IDK Rate: among unanswerable questions, % where model said IDK
    idk_rate = unanswerable["model_said_idk"].mean() * 100 if n_unanswerable > 0 else 0.0

    # Hallucination Rate: among unanswerable questions, % where model gave a confident answer
    hallucination_rate = (~unanswerable["model_said_idk"]).mean() * 100 if n_unanswerable > 0 else 0.0

    # Accuracy on answerable questions only
    acc_answerable = answerable["gpt_correct"].mean() * 100 if n_answerable > 0 else 0.0

    # Per-type breakdown of hallucination rate
    type_breakdown = {}
    for qtype, group in unanswerable.groupby("type"):
        type_breakdown[qtype] = {
            "idk_rate": group["model_said_idk"].mean() * 100,
            "hallucination_rate": (~group["model_said_idk"]).mean() * 100,
            "n": len(group),
        }

    return {
        "model": model_name,
        "total_questions": total,
        "n_unanswerable": n_unanswerable,
        "n_answerable": n_answerable,
        "gpt_score": gpt_score,
        "paper_score": PAPER_ACCURACY.get(model_name, float("nan")),
        "idk_rate": idk_rate,
        "hallucination_rate": hallucination_rate,
        "acc_answerable": acc_answerable,
        "type_breakdown": type_breakdown,
        "result_df": result_df,
    }


def main(args):
    results_dir = Path(args.results_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} questions from {args.csv}")
    print(f"Unanswerable (any GT = IDK): {df.apply(is_unanswerable_gt, axis=1).sum()}\n")

    # Analyze each model
    all_metrics = []
    all_result_dfs = {}
    for model_name in MODEL_NAMES:
        print(f"Analyzing {model_name}...")
        metrics = analyze_model(model_name, df, results_dir, pred_dir)
        if metrics:
            all_result_dfs[model_name] = metrics.pop("result_df")
            all_metrics.append(metrics)

    if not all_metrics:
        print("No results found. Run eval.py first.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            "Model": m["model"],
            "GPT Score (%)": f"{m['gpt_score']:.1f}",
            "Paper Score (%)": f"{m['paper_score']:.1f}",
            "IDK Rate (%)": f"{m['idk_rate']:.1f}",
            "Hallucination Rate (%)": f"{m['hallucination_rate']:.1f}",
            "Acc Answerable (%)": f"{m['acc_answerable']:.1f}",
            "N Unanswerable": m["n_unanswerable"],
        })

    print("\n=== Overall Summary ===")
    print(tabulate(summary_rows, headers="keys", tablefmt="github"))

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "hallucination_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # ── Per-type hallucination table ───────────────────────────────────────────
    # Collect all question types
    all_types = set()
    for m in all_metrics:
        all_types.update(m["type_breakdown"].keys())

    type_rows = []
    for m in all_metrics:
        row = {"Model": m["model"]}
        for qtype in sorted(all_types):
            tb = m["type_breakdown"].get(qtype, {})
            hall = tb.get("hallucination_rate", float("nan"))
            n = tb.get("n", 0)
            row[f"{qtype} (hall%, n={n})"] = f"{hall:.1f}" if n > 0 else "N/A"
        type_rows.append(row)

    print("\n=== Hallucination Rate by Question Type (unanswerable Qs only) ===")
    print(tabulate(type_rows, headers="keys", tablefmt="github"))

    type_df = pd.DataFrame(type_rows)
    type_path = output_dir / "hallucination_by_type.csv"
    type_df.to_csv(type_path, index=False)
    print(f"Saved type breakdown to {type_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/evaluations",
                        help="Directory containing results_{model}.json files from eval.py")
    parser.add_argument("--pred_dir", default="outputs/predictions",
                        help="Directory containing {model}.jsonl prediction files")
    parser.add_argument("--csv", default="data/test_half_release.csv")
    parser.add_argument("--output_dir", default="outputs/final",
                        help="Directory to write summary CSVs")
    args = parser.parse_args()
    main(args)
