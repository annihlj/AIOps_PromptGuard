# PromptGuard

![Python 3.14](https://img.shields.io/badge/python-3.14-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

PromptGuard checks employee prompts for sensitive information **before** they are sent to an external
AI service. It detects personal data and credentials and returns one of three decisions:

| Decision | Meaning | Example |
|---|---|---|
| **allow** | No relevant sensitive information | "Summarise the meeting notes in 5 bullet points." |
| **warn** | Personal data: review or edit the prompt before sending | a name, e-mail address, phone number or postal address |
| **block** | Secrets or financial / government identifiers | a password, PIN, IBAN or credit card number |

**Goal:** help IT security and compliance teams reduce accidental disclosure of personal data and
credentials, with a focus on **German** companies.

## Table of contents

1. [Project approach](#project-approach)
2. [Current status and results](#current-status-and-results)
3. [Repository structure](#repository-structure)
4. [Getting started](#getting-started)
5. [Data, policy and split](#data-policy-and-split)
6. [Rules for training and evaluation](#rules-for-training-and-evaluation)
7. [Experiment tracking with MLflow](#experiment-tracking-with-mlflow)
8. [Troubleshooting](#troubleshooting)
9. [License and data sources](#license-and-data-sources)

## Project approach

1. **Data:** prepare the multilingual PII dataset `ai4privacy/pii-masking-200k` (de, en, fr, it) and map
   every PII label to allow / warn / block with a policy file.
2. **Baselines:** a regex baseline and a TF-IDF + Logistic Regression classifier.
3. **Better models:** compare further classifiers and a small Named Entity Recognition (NER) model.
4. **Tracking:** log every experiment with **MLflow** on a shared server.
5. **Deployment:** expose the selected model through a **FastAPI** service.

We compare models on detection quality (especially missed block prompts), false alarms on harmless
prompts and response time.

**Project targets** (on the validation set):

| Metric | Target |
|---|---|
| Recall block (share of block prompts that are blocked) | ≥ 95 % |
| False block rate on harmless (allow) prompts | ≤ 5 % |
| Response time (95th percentile) | < 100 ms |

**Tech stack:** Python 3.14 · Jupyter · scikit-learn · Hugging Face · MLflow (DagsHub) · FastAPI · GitHub

## Current status and results

| Milestone | Content | Status |
|---|---|---|
| M1 | Dataset, policy, fixed train / validation / test split | done |
| M2 A | Baselines (regex, TF-IDF + LogReg), tuning, harmless prompts | done |
| M2 B / C | Further classifiers, NER model | in progress |
| M3 – M5 | API, final evaluation on the test set | planned |

Results on the ai4privacy validation set (31,380 texts, all languages):

| Model | Notebook | Macro F1 | Recall block | Recall block (de) | Latency p95 |
|---|---|---|---|---|---|
| Regex baseline | [`baseline_regex.ipynb`](notebooks/baseline_regex.ipynb) | 0.222 | 0.126 | 0.131 | 0.08 ms |
| TF-IDF + LogReg baseline | [`baseline_tfidf_logreg.ipynb`](notebooks/baseline_tfidf_logreg.ipynb) | 0.870 | 0.918 | 0.912 | 1.3 ms |
| TF-IDF + tuned (C = 30, block threshold 0.2) | [`tfidf_tuning.ipynb`](notebooks/tfidf_tuning.ipynb) | 0.864 | **0.955** | 0.953 | 2.9 ms |
| TF-IDF + harmless prompts in training | [`tfidf_harmless.ipynb`](notebooks/tfidf_harmless.ipynb) | 0.844 | **0.953** | 0.944 | 2.8 ms |
| Linear SVM tuned (C = 1.0, threshold -0.4) | [`svm_linear.ipynb`](notebooks/svm_linear.ipynb) | 0.865 | **0.956** | 0.955 | 3.4 ms |
| Linear SVM + harmless prompts | [`svm_linear.ipynb`](notebooks/svm_linear.ipynb) | 0.862 | **0.957** | 0.956 | 3.5 ms |

- The last four rows come from one comparison (chapter 9 of `svm_linear.ipynb`, saved in
`reports/model_comparison_m2.csv`): same data, same code, latency measured on the same laptop.
The baseline latencies were measured on another laptop and are only roughly comparable.
- Adding harmless prompts raised the share of correctly allowed real harmless prompts from 41% to 93%,
and accuracy on our own 200 German test prompts from 65% to 78%.
- The downside: 20 of 60 warn prompts with names or e-mail addresses are now let through (before: 1),
because the model links the short request style to "allow". Details and next step (plan B):
conclusion of `tfidf_harmless.ipynb`.

Details: [`reports/`](reports/) and the conclusion section of each notebook.

## Repository structure

```text
AIOps_PromptGuard/
├── app/                    FastAPI service (planned)
├── config/
│   └── policy.yaml         PII label → allow / warn / block
├── data/                   datasets (only splits.csv and own_prompts.csv are in Git)
│   ├── splits.csv          fixed train / validation / test split
│   └── own_prompts.csv     200 hand-written German test prompts
├── models/                 trained models (*.joblib, not in Git)
├── notebooks/              data preparation and experiments
├── reports/                result tables and data checks
├── scripts/                one-off analysis scripts (e.g. check_harmless.py)
├── src/                    shared Python code
│   ├── data_utils.py       load splits, policy, fingerprints
│   ├── download_data.py    download ai4privacy dataset
│   ├── harmless_data.py    build harmless prompts (oasst1 + templates)
│   ├── make_splits.py      create / verify the split
│   └── check_mlflow.py     test the connection to the MLflow server
├── tests/                  unit tests (planned)
├── .env.example            template for the MLflow credentials
├── .python-version         Python version of the project (3.14)
├── requirements.txt        Python packages
└── LICENSE
```

## Getting started

Every team member creates their **own** virtual environment (`.venv`) on their computer. It keeps the
packages of this project separate from other projects. The `.venv` folder is never committed to Git.

The steps work on **Windows** and **macOS**. Where the commands differ, both versions are shown.

### 1. Install the prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | **3.14** (any 3.14.x) | https://www.python.org/downloads/ |
| Git | any recent version | https://git-scm.com/downloads |
| VS Code | with the **Python** and **Jupyter** extensions (Microsoft) | https://code.visualstudio.com/ |

> **Why exactly Python 3.14?** Different Python and package versions can give slightly different
> model results. With the same version, everybody gets the same numbers from the same notebook.

**Windows:** run the Python installer from python.org. In the installer, keep the option for the
**`py` launcher** enabled. Then check in a new terminal (PowerShell):

```powershell
py -3.14 --version
```

**macOS:** run the Python installer from python.org (or `brew install python@3.14`). Then check:

```bash
python3.14 --version
```

Both commands should print `Python 3.14.x`.

### 2. Get the project

Clone the repository (only once) and open a terminal in the project folder:

```bash
git clone https://github.com/annihlj/AIOps_PromptGuard.git
cd AIOps_PromptGuard
```

If you already have the project, get the latest version instead:

```bash
git pull
```

### 3. Create the virtual environment

Run this once, in the project folder:

**Windows (PowerShell):**

```powershell
py -3.14 -m venv .venv
```

**macOS:**

```bash
python3.14 -m venv .venv
```

This creates the folder `.venv` inside the project folder.

> **Already have a `.venv` with another Python version?** Delete it and create it again:
> close VS Code and all notebooks first, then
> Windows: `Remove-Item -Recurse -Force .venv` · macOS: `rm -rf .venv`
> and run the command above. Afterwards continue with steps 4 – 6.

### 4. Activate the virtual environment

Do this in **every new terminal** you use for this project.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate.bat
```

**macOS:**

```bash
source .venv/bin/activate
```

Your terminal prompt now starts with `(.venv)`. Check that the right Python is used:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The first line must show `Python 3.14.x`, the second a path **inside** the project's `.venv` folder.

> From here on, Windows and macOS use the same `python` commands, as long as the environment is active.

### 5. Install the packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the second command again whenever `requirements.txt` changes (for example after a `git pull`).

### 6. Register the environment as a notebook kernel

A kernel is the Python process that runs the code cells of a notebook. Register the environment once
(and again after you re-create `.venv`):

```bash
python -m ipykernel install --user --name promptguard --display-name "Python 3.14 (PromptGuard)"
```

### 7. Download and prepare the data

The data files are not stored in Git. Download and build them once, in this order:

```bash
python src/download_data.py      # ai4privacy dataset into data/ (files that exist are skipped)
python src/make_splits.py        # only checks the existing split
python src/harmless_data.py      # harmless prompts into data/harmless_prompts.csv (downloads oasst1, ~40 MB)
python src/short_prompts.py      # short prompts into data/short_prompts.csv
```

Each script prints a fingerprint. It must match the value below, otherwise your data is different from the
team version: tell the team before training a model.

| Script | Expected fingerprint |
|---|---|
| `make_splits.py` | `6afd6a257de1` |
| `harmless_data.py` | `32d24c900fd7` |
| `short_prompts.py` | `bb97fd4d3e90` |

### 8. Connect to the shared MLflow server (DagsHub)

All MLflow runs are stored on a shared server on [DagsHub](https://dagshub.com), so the whole team sees
the same experiments. Without this step the notebooks still work, but the runs are only saved locally
(`mlflow.db`).

1. Create a free DagsHub account (sign in with GitHub) and ask the repo owner to add you as a
   collaborator.
2. In DagsHub, open **Settings → Tokens** and copy your access token.
3. **Copy** (do not rename) `.env.example` to `.env` and fill in your DagsHub user name and token.
   `.env` is in `.gitignore`. **Never** put the token into a notebook or into `.env.example`.
4. Test the connection:

   ```bash
   python src/check_mlflow.py
   ```

   Expected output: `Connection OK.`

### 9. Run the notebooks

1. Open the **project folder** in VS Code (*File → Open Folder*).
2. Open a notebook in `notebooks/`.
3. Click **Select Kernel** (top right) → **Jupyter Kernel** → **Python 3.14 (PromptGuard)**.
   Alternatively choose **Python Environments** → the project's `.venv`.
4. Run the cells from top to bottom (**Run All**, or `Shift + Enter` cell by cell).

> Check the kernel for **each** notebook. Activating `.venv` in the terminal does not select it as
> the notebook kernel.

Run the experiment notebooks in this order, because later notebooks load models from earlier ones:

1. `preperation.ipynb` – data analysis
2. `baseline_regex.ipynb`
3. `baseline_tfidf_logreg.ipynb`
4. `tfidf_tuning.ipynb` (saves `models/tfidf_logreg_tuned.joblib`)
5. `tfidf_harmless.ipynb` (needs the model from step 4)

### 10. End your session

Save the notebooks, then deactivate the environment in the terminal:

```bash
deactivate
```

## Data, policy and split

### Dataset

[`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k): the German,
English, French and Italian files (209,261 synthetic texts, 56 PII labels). The first analysis is in
[`notebooks/preperation.ipynb`](notebooks/preperation.ipynb).

### Policy: PII label → allow / warn / block

[`config/policy.yaml`](config/policy.yaml) maps every PII label to a decision (version 0.1, proposal):

- **block:** credentials and financial / government identifiers (password, PIN, credit card, IBAN,
  account number, SSN, ...)
- **warn:** data that identifies or locates a person (names, e-mail, phone, address, date of birth,
  IP, ...)
- **allow:** context data that does not identify a person on its own (date, amount, job title, city, ...)

A text gets the **most severe** decision of all its PII entities (block > warn > allow).
Result on the full dataset: 7.8 % allow, 60.7 % warn, 31.4 % block.

### Train / validation / test split

All models use the **same fixed split**, stored in [`data/splits.csv`](data/splits.csv)
(columns `id`, `split`).

| Rule | Value |
|---|---|
| Shares | 70 % train / 15 % validation / 15 % test |
| Stratified by | language × policy decision (e.g. `de_allow`, `fr_block`) |
| Grouped by | masked template (`target_text`): texts that share a template are always in the same split, so the test set contains no templates seen in training |
| Excluded | 41 lorem-ipsum filler texts (`split = excluded`) |
| Seed | 42 |
| Fingerprint | `6afd6a257de1` (SHA-256 of `splits.csv`, first 12 characters) |

Resulting sizes: train 146,455 · validation 31,380 · test 31,385. Every language × decision group has
the same share in all three splits.

The split was created with `python src/make_splits.py`. The script is deterministic and refuses to
overwrite an existing `splits.csv` unless `--force` is passed. **Do not re-create the split** unless the
whole team agrees, otherwise results of different models are no longer comparable. The split is
independent of the policy: if labels in `policy.yaml` move, only the `decision` column changes.

### Harmless prompts and own test prompts

The ai4privacy dataset contains **no text without personal data**, so a model trained only on it never
sees a normal, harmless business prompt. We therefore use two extra files:

| File | Content | Used for |
|---|---|---|
| `data/harmless_prompts.csv` (built by `src/harmless_data.py`, not in Git) | PII-free prompts with decision `allow`: first user questions from [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (de/en/fr/it, filtered for personal data) + company-style template prompts | oasst1: 70 / 15 / 15 train / validation / test (deterministic, by conversation). Templates: train only |
| `data/own_prompts.csv` (in Git) | 200 hand-written realistic German prompts: 80 allow, 60 warn, 60 block, column `style` = formal / casual | **evaluation only, never train or tune on them** |

Prompts with person names are removed, because names are `warn` under policy v0.1 and the file may only
contain `allow` prompts. Removed prompts and the reason are written to `data/harmless_removed.csv`
(not in Git). Spot check of 50 kept prompts: 2 policy errors found (famous people: Charles de Gaulle, Shakespeare).

A quality check of the harmless prompts (private names, number filter, overlap with our test prompts,
distribution) is in [`reports/harmless_check.md`](reports/harmless_check.md)
(script: `scripts/check_harmless.py`).

### Short prompts with and without personal data (data fix)

With the harmless prompts, the models let 15 to 20 of 60 warn prompts through. The reason is the writing style:
all ai4privacy texts (warn / block) are long documents (median ~190 characters), all harmless prompts (allow)
are short requests (~70 characters). The models learned "short prompt = allow".

`src/short_prompts.py` builds short prompts in German and English for **all three classes** from the same
sentence frames (e.g. "Schreib eine Mail an {recipient} zum Thema {topic}."). Only the slot content decides:
"das Vertriebsteam" = allow, a name or e-mail = warn, an IBAN or password = block. The decision is calculated
with our policy (`prompt_decision`), not set by hand.

| Split | Frames | Prompts | Personal data values from |
|---|---|---|---|
| train | 14 per language | 2,000 per class and language (12,000) | ai4privacy train |
| validation | 6 other frames per language | 100 per class and language (200 / 200 / 200) | ai4privacy validation |
| test | 6 other frames per language | 100 per class and language (200 / 200 / 200) | ai4privacy test |

Validation and test use other sentence frames than training, so a good result there means the model learned
the content, not the frames. The frames were written without looking at `data/own_prompts.csv`.
The file `data/short_prompts.csv` is built by the script and not in Git.


### Loading the data in code

Load the data **only** through [`src/data_utils.py`](src/data_utils.py). From a notebook in `notebooks/`:

```python
import sys
sys.path.append("../src")
from data_utils import load_all_splits, split_fingerprint
from harmless_data import load_harmless, harmless_fingerprint

splits = load_all_splits()               # dict with "train", "validation", "test"
train, val = splits["train"], splits["validation"]
harmless_train = load_harmless("train")
print(split_fingerprint(), harmless_fingerprint())
```

Each split is a DataFrame with all original dataset columns plus `decision`:

| Column | Use it for |
|---|---|
| `source_text` | model input (the prompt) |
| `decision` | target for classifiers (allow / warn / block) |
| `privacy_mask` | character spans + labels, for regex / span-based evaluation |
| `mbert_text_tokens`, `mbert_bio_labels` | token-level BIO labels for NER models |
| `language` | per-language evaluation (especially `de`) |

Other helpers: `load_split("train")` loads a single split; `load_policy()` and
`prompt_decision(privacy_mask, policy)` turn detected PII labels into a decision. Use the same function
to turn NER predictions into allow / warn / block.

## Rules for training and evaluation

1. Train only on `train`.
2. Tune hyperparameters and thresholds and compare models only on `validation`.
3. Evaluate on `test` **once**, at the end, for the final models (Milestone 5).
4. `data/own_prompts.csv` is an extra test set: only look at the results, never tune on them.
5. Log `split_fingerprint()`, `harmless_fingerprint()` and the policy version with every MLflow run.
6. Report results overall **and** for `language == "de"`.
7. Use Python 3.14 and `requirements.txt`, so results are reproducible across the team.

## Experiment tracking with MLflow

- All notebooks log to the shared DagsHub server when `.env` is set up (see
  [step 8](#8-connect-to-the-shared-mlflow-server-dagshub)), otherwise to a local `mlflow.db`.
- All runs belong to the experiment **`promptguard`**. Each run logs parameters, validation metrics and
  the fingerprints of the data it used.
- View the runs in DagsHub under **Experiments**, or locally with
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Troubleshooting

| Problem | Solution |
|---|---|
| `py` is not found (Windows) | Re-run the Python installer and enable the `py` launcher, or use the full path to `python.exe` of Python 3.14. |
| PowerShell: *"running scripts is disabled on this system"* when activating | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `python --version` does not show 3.14 inside `.venv` | The environment was created with another Python. Re-create it (step 3). |
| `ModuleNotFoundError` in a notebook | The notebook uses the wrong kernel. Select **Python 3.14 (PromptGuard)** (step 9). |
| `FileNotFoundError` for a data file | Run the commands in step 7. |
| `tfidf_harmless.ipynb` stops because a model is missing | Run `tfidf_tuning.ipynb` first. |
| `check_mlflow.py`: `MLFLOW_TRACKING_URI is not set` | `.env` is missing or empty (step 8). |

## License and data sources

The project's own code and data (e.g. `src/`, `scripts/`, `notebooks/`, `app/`, `config/policy.yaml`,
`data/own_prompts.csv`, `data/splits.csv`) are licensed under the [MIT License](LICENSE).

Third-party data is **not** part of this repository (`data/*` is in `.gitignore`). It is downloaded by
our scripts and keeps its own license:

| Dataset | Used for | License | Downloaded by |
|---|---|---|---|
| [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | main training / validation / test data | Custom Ai4Privacy license ([license.md](https://huggingface.co/datasets/ai4privacy/pii-masking-200k/blob/main/license.md)): free for individuals, organisations with up to 3 staff and non-profits; derivatives must not be distributed, sold, re-licensed or sublicensed. Copyright © 2023 Ai4Privacy | `src/download_data.py` |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | harmless prompts | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | `src/harmless_data.py` |

- oasst1 citation: Köpf et al. (2023), "OpenAssistant Conversations: Democratizing Large Language Model
  Alignment", arXiv:2304.07327. We only use the first user message of each conversation and filter it
  (see `src/harmless_data.py`).
- `data/splits.csv` only contains row ids and split names, no texts from ai4privacy.
- Models trained on ai4privacy (e.g. `models/*.joblib`, MLflow artifacts) are covered by the ai4privacy
  terms, not by our MIT license. Do not publish them without checking those terms.
- A productive use of PromptGuard in a company with more than three employees would need a commercial
  Ai4Privacy license or retraining on freely licensed data.
