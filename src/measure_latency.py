"""Measure the response time of the encoder model on THIS computer (CPU, one prompt at a time).

WHY THIS SCRIPT EXISTS
----------------------
The encoder is trained in Google Colab on a GPU. But PromptGuard will run on a normal
server without GPU, and our TF-IDF / SVM times (about 3 ms, p95) were measured on a
laptop. To compare fairly, we measure the encoder in the same way:
    - on the CPU,
    - one prompt at a time (batch size 1), like in the real application,
    - the same 1,000 validation prompts as in svm_linear.ipynb chapter 9
      (val["source_text"].sample(1000, random_state=42)),
    - the time includes tokenizing + model + block rule (like pipeline.predict for TF-IDF).

USAGE (from the project folder, venv activated, after
       python -m pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu)

    # 1) download the trained model from MLflow (DagsHub) into models/xlmr_lora_benign/ and measure
    python src/measure_latency.py --download

    # 2) same, and write the result into the MLflow run (latency_p50_ms, latency_p95_ms)
    python src/measure_latency.py --download --log

    # the model folder already exists (e.g. copied from Google Drive):
    python src/measure_latency.py --model-dir models/xlmr_lora_benign --log

Close other programs while measuring, otherwise the times are too high.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR / "src"))

SEED = 42
DEFAULT_RUN_NAME = "xlmr_lora_benign"


def connect_mlflow():
    """Same MLflow setup as in the notebooks: shared DagsHub server from .env."""
    import mlflow
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise SystemExit("MLFLOW_TRACKING_URI is missing in .env (see .env.example).")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("promptguard")
    return mlflow


def find_run_id(mlflow, run_name: str) -> str:
    """The newest run with this name (same approach as svm_linear.ipynb chapter 9)."""
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"No MLflow run with the name {run_name!r} found.")
    return runs.loc[0, "run_id"]


def latency_prompts(n: int) -> list[str]:
    """The same validation sample as in svm_linear.ipynb chapter 9 (ai4privacy val + harmless val)."""
    from data_utils import load_all_splits
    from harmless_data import load_harmless

    columns = ["source_text", "decision", "language"]
    pii_val = load_all_splits()["validation"]
    val = pd.concat([pii_val[columns].assign(origin="ai4privacy"),
                     load_harmless("validation")[columns].assign(origin="harmless")],
                    ignore_index=True)
    return val["source_text"].sample(n, random_state=SEED).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(PROJECT_DIR / "models" / DEFAULT_RUN_NAME))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="MLflow run to download from / log to")
    parser.add_argument("--download", action="store_true", help="download the model from MLflow first")
    parser.add_argument("--log", action="store_true", help="log the result to the MLflow run")
    parser.add_argument("--n", type=int, default=1000, help="number of prompts to measure")
    args = parser.parse_args()

    import torch
    from encoder_utils import load_encoder, predict_proba, predict_with_threshold

    model_dir = Path(args.model_dir)
    mlflow, run_id = None, None
    if args.download or args.log:
        mlflow = connect_mlflow()
        run_id = find_run_id(mlflow, args.run_name)
    if args.download:
        # The notebook logs the model folder as the artifact "model"
        print(f"Downloading the model of run {run_id} ...")
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model",
                                            dst_path=str(model_dir.parent))
        downloaded = model_dir.parent / "model"
        if downloaded != model_dir:
            if model_dir.exists():
                raise SystemExit(f"{model_dir} already exists. Delete it or run without --download.")
            downloaded.rename(model_dir)
    if not model_dir.exists():
        raise SystemExit(f"{model_dir} not found. Use --download or --model-dir.")

    # CPU only, so the numbers are comparable with TF-IDF / SVM
    model, tokenizer, cfg = load_encoder(model_dir, device="cpu", merge=True)
    threshold = cfg["block_threshold"]
    print(f"Model: {cfg['base_model']} + LoRA | threshold: {threshold} | "
          f"torch threads: {torch.get_num_threads()}")

    print("Loading the validation prompts (takes a moment) ...")
    prompts = latency_prompts(args.n)

    def classify(prompt: str) -> str:
        proba = predict_proba(model, tokenizer, [prompt], cfg["max_length"], batch_size=1)
        return predict_with_threshold(proba, cfg["classes"], threshold)[0]

    # Warm-up: the first calls are always slower (memory is prepared), we do not count them
    for prompt in prompts[:20]:
        classify(prompt)

    times = []
    for prompt in prompts:
        start = time.perf_counter()
        classify(prompt)
        times.append((time.perf_counter() - start) * 1000)     # milliseconds

    result = {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_mean_ms": float(np.mean(times)),
    }
    cpu = platform.processor() or platform.machine()
    print(f"\n{len(times)} prompts, CPU: {cpu}")
    for key, value in result.items():
        print(f"  {key:<16} {value:7.1f}")
    print("For comparison (reports/model_comparison_m2.csv, same laptop): SVM + harmless p95 = 3.5 ms")

    if args.log:
        with mlflow.start_run(run_id=run_id):        # reopens the existing run
            mlflow.log_metrics(result)
            # tags (not params), so the measurement can be repeated on another laptop
            mlflow.set_tags({"latency_cpu": cpu, "latency_torch_threads": str(torch.get_num_threads()),
                             "latency_n_prompts": str(len(times))})
        print(f"Logged to MLflow run {args.run_name} ({run_id}).")


if __name__ == "__main__":
    main()
