"""Check the connection to our shared MLflow server.

Run from the project folder (venv activated):
    python src/check_mlflow.py

It reads .env, connects to MLFLOW_TRACKING_URI and lists the runs of the
experiment "promptguard". Nothing is written to the server.
"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

uri = os.getenv("MLFLOW_TRACKING_URI")
if not uri:
    sys.exit("MLFLOW_TRACKING_URI is not set. Copy .env.example to .env and fill it in.")
for var in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
    if not os.getenv(var) or os.getenv(var, "").startswith("<"):
        sys.exit(f"{var} is missing in .env.")

mlflow.set_tracking_uri(uri)
print("Tracking URI:", uri)

try:
    experiment = mlflow.get_experiment_by_name("promptguard")
except Exception as error:  # wrong URL, wrong token, no internet ...
    sys.exit(f"Could not connect: {error}")

print("Connection OK.")
if experiment is None:
    print('Experiment "promptguard" does not exist yet. It is created by the first notebook run.')
else:
    runs = mlflow.search_runs([experiment.experiment_id], output_format="list")
    print(f'Experiment "promptguard": {len(runs)} run(s)')
    for run in runs:
        print(" -", run.info.run_name, run.info.status)
