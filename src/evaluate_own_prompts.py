"""Evaluate the saved models on our own German test prompts, per version (v1 / v2 / all).

WHY THIS SCRIPT EXISTS
----------------------
data/own_prompts.csv was extended (src/own_prompts.py, Olga):
    v1 = the original 200 prompts (80 allow / 60 warn / 60 block); all earlier runs were measured on these
    v2 = 400 new prompts (120 allow / 140 warn / 140 block)
    all = 600 prompts, 200 per class
Own prompts are ONLY used for evaluation, never for training. So the models do not have to be
trained again: we download them from MLflow (DagsHub) and measure them on the new prompts.

WHAT IT DOES
------------
1. Downloads each model of MODELS from its MLflow run into models/ (if it is not there yet).
2. Predicts all 600 prompts on the CPU, with the block threshold saved with the model.
3. Calculates the same metrics as the notebooks, for v1, v2 and all.
   Check: the old "own_prompts_..." metrics of the run must be reproduced exactly (on v1, or on all
   600 for runs logged after the extension). If not, the model was not loaded correctly and
   nothing is logged.
4. Saves reports/own_prompts_v2_comparison.csv and reports/own_prompts_v2_wrong.csv.
5. With --log: adds own_prompts_v1_*, own_prompts_v2_* and own_prompts_all_* to each existing run.
   These names mean the same in every run. The old own_prompts_* metrics are not changed
   (careful: depending on the run they were measured on v1 or on all 600 prompts).

USAGE (from the project folder, venv activated)
    python src/evaluate_own_prompts.py          # measure and save the CSV files, nothing logged
    python src/evaluate_own_prompts.py --log    # the same + write the new metrics to MLflow
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

# MLflow prints an emoji when a run ends; the Windows console cannot show it without UTF-8
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR / "src"))
from encoder_utils import load_encoder, predict_proba, predict_with_threshold  # noqa: E402

MODEL_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# column name in the table -> (MLflow run name, model type, artifact)
MODELS = {
    "SVM + harmless": ("svm_harmless", "sklearn", "svm_harmless.joblib"),
    "SVM + short prompts": ("svm_short", "sklearn", "svm_short.joblib"),
    "Encoder run 1 (+ harmless)": ("xlmr_lora_benign", "encoder", "model"),
    "Encoder run 2 (+ harmless + short prompts)": ("xlmr_lora_benign_templates", "encoder", "model"),
}
# metrics that are compared with the values already in MLflow (check that the model loaded correctly)
CHECK_METRICS = ["accuracy", "recall_block", "false_block_allow", "warn_to_allow"]


# ------------------------------------------------------------------ metrics (same as in the notebooks)
def evaluate(y_true, y_pred) -> dict:
    """Same function as in svm_linear.ipynb / train_encoder_lora_colab.ipynb."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    is_allow, is_warn = y_true == "allow", y_true == "warn"
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "recall_block": recall_score(y_true, y_pred, labels=["block"], average="macro", zero_division=0),
        "recall_warn": recall_score(y_true, y_pred, labels=["warn"], average="macro", zero_division=0),
        "recall_allow": recall_score(y_true, y_pred, labels=["allow"], average="macro", zero_division=0),
        "false_block_allow": float((y_pred[is_allow] == "block").mean()) if is_allow.any() else np.nan,
        "false_block_warn": float((y_pred[is_warn] == "block").mean()) if is_warn.any() else np.nan,
    }


def own_metrics(expected, pred, casual) -> dict:
    """The own_prompts_* metrics of the notebooks."""
    m = evaluate(expected, pred)
    m["accuracy"] = float((pred == expected).mean())
    m["warn_to_allow"] = float((pred[expected == "warn"] == "allow").mean())
    m["casual_recall_block"] = float((pred[(expected == "block") & casual] == "block").mean())
    return m


def table_rows(expected, pred, casual) -> dict:
    """The "own: ..." rows of the comparison tables (counts instead of shares)."""
    def count(hit, total):
        return f"{int(hit.sum())} of {int(total.sum())}"
    block, warn, allow = expected == "block", expected == "warn", expected == "allow"
    return {
        "correct": count(pred == expected, expected == expected),
        "macro F1": round(f1_score(expected, pred, average="macro"), 3),
        "block prompts caught": count(pred[block] == "block", block),
        "casual block prompts caught": count(pred[block & casual] == "block", block & casual),
        "warn prompts caught (warn)": count(pred[warn] == "warn", warn),
        "warn prompts let through (allow)": count(pred[warn] == "allow", warn),
        "harmless prompts allowed": count(pred[allow] == "allow", allow),
        "harmless prompts blocked": count(pred[allow] == "block", allow),
    }


# ------------------------------------------------------------------ models
def connect_mlflow():
    import mlflow
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is missing in .env (see .env.example).")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("promptguard")
    return mlflow


def newest_run(mlflow, run_name: str):
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"No MLflow run with the name {run_name!r}.")
    return runs.iloc[0]


def predict(mlflow, run_id: str, run_name: str, kind: str, artifact: str, prompts: list[str]) -> np.ndarray:
    """Download the model of one run (once) and return its decisions for the prompts."""
    if kind == "sklearn":
        path = MODEL_DIR / artifact
        if not path.exists():
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact, dst_path=str(MODEL_DIR))
        bundle = joblib.load(path)
        pipe = bundle["pipeline"]
        # SVM -> decision_function (scores), Logistic Regression -> predict_proba (same as the notebooks)
        if hasattr(pipe, "decision_function") and not hasattr(pipe, "predict_proba"):
            scores = pipe.decision_function(prompts)
        else:
            scores = pipe.predict_proba(prompts)
        return predict_with_threshold(scores, pipe.classes_, bundle["block_threshold"])

    path = MODEL_DIR / run_name                  # e.g. models/xlmr_lora_benign/
    if not path.exists():
        downloaded = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact,
                                                              dst_path=str(MODEL_DIR / f"{run_name}_download")))
        downloaded.rename(path)
        (MODEL_DIR / f"{run_name}_download").rmdir()
    model, tokenizer, cfg = load_encoder(path, device="cpu", merge=True)
    proba = predict_proba(model, tokenizer, prompts, cfg["max_length"])
    return predict_with_threshold(proba, cfg["classes"], cfg["block_threshold"])


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", action="store_true", help="add the new metrics to the existing MLflow runs")
    args = parser.parse_args()

    own = pd.read_csv(PROJECT_DIR / "data" / "own_prompts.csv")
    if "version" not in own.columns:
        raise SystemExit("data/own_prompts.csv has no column 'version' (git pull for Olga's v2 prompts).")
    expected = own["expected_decision"].to_numpy()
    casual = (own["style"] == "casual").to_numpy()
    subsets = {"v1": (own["version"] == "v1").to_numpy(),
               "v2": (own["version"] == "v2").to_numpy(),
               "all": np.ones(len(own), dtype=bool)}
    print({name: int(mask.sum()) for name, mask in subsets.items()}, "own prompts")

    mlflow = connect_mlflow()
    columns, wrong, to_log, problems = {}, [], {}, []
    for column, (run_name, kind, artifact) in MODELS.items():
        run = newest_run(mlflow, run_name)
        print(f"\n{column}: run {run_name} ({run.run_id[:8]})")
        pred = predict(mlflow, run.run_id, run_name, kind, artifact, own["prompt"].tolist())

        metrics = {name: own_metrics(expected[m], pred[m], casual[m]) for name, m in subsets.items()}
        # Check that the model loaded correctly: the old own_prompts_* metrics of the run must be
        # reproduced exactly. Most runs measured them on v1 (200); runs logged after Olga's
        # extension measured them on all 600 prompts. So we accept v1 OR all and say which one.
        logged = {key: run.get(f"metrics.own_prompts_{key}") for key in CHECK_METRICS}
        logged = {k: v for k, v in logged.items() if v is not None and not pd.isna(v)}
        matches = [name for name in ("v1", "all")
                   if logged and all(abs(v - metrics[name][k]) < 1e-9 for k, v in logged.items())]
        if logged and not matches:
            problems.append(f"{column}: own_prompts_* in MLflow {logged} match neither v1 nor all")
        print("  check against MLflow:", f"OK, its own_prompts_* were measured on {matches[0]}" if matches
              else ("no own_prompts_* in MLflow" if not logged else "DIFFERENT"))

        for name, m in subsets.items():
            for row, value in table_rows(expected[m], pred[m], casual[m]).items():
                columns.setdefault(column, {})[f"own {name}: {row}"] = value
        # explicit names that mean the same in every run: own_prompts_v1_*, own_prompts_v2_*, own_prompts_all_*
        to_log[run.run_id] = {f"own_prompts_{name}_{k}": v for name in subsets for k, v in metrics[name].items()}
        bad = pred != expected
        wrong.append(own[bad].assign(model=column, predicted=pred[bad]))

    comparison = pd.DataFrame(columns)
    REPORTS_DIR.mkdir(exist_ok=True)
    comparison.to_csv(REPORTS_DIR / "own_prompts_v2_comparison.csv")
    wrong_df = pd.concat(wrong, ignore_index=True)[["model", "version", "id", "prompt", "expected_decision",
                                                     "predicted", "category", "style"]]
    wrong_df.to_csv(REPORTS_DIR / "own_prompts_v2_wrong.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 10):
        print("\n", comparison)
    print("\nSaved: reports/own_prompts_v2_comparison.csv, reports/own_prompts_v2_wrong.csv")

    if problems:
        print("\nNOT logged, the v1 numbers differ from MLflow:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    if args.log:
        for run_id, metrics in to_log.items():
            with mlflow.start_run(run_id=run_id):      # reopens the existing run, adds the new metrics
                mlflow.log_metrics(metrics)
        print(f"Logged own_prompts_v1_*, own_prompts_v2_* and own_prompts_all_* to {len(to_log)} MLflow runs.")
    else:
        print("Nothing logged to MLflow (run again with --log).")


if __name__ == "__main__":
    main()
