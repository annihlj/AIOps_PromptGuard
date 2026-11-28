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
3. **Better models:** compare further classifiers (linear SVM), a pre-trained multilingual encoder
   (XLM-RoBERTa) fine-tuned with LoRA, and a small Named Entity Recognition (NER) model.
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

**Tech stack:** Python 3.14 · Jupyter · scikit-learn · PyTorch · Hugging Face Transformers · PEFT (LoRA) ·
Google Colab (GPU training) · MLflow (DagsHub) · FastAPI · GitHub

## Current status and results

| Milestone | Content | Status |
|---|---|---|
| M1 | Dataset, policy, fixed train / validation / test split | done |
| M2 A | Baselines (regex, TF-IDF + LogReg), tuning, harmless prompts | done |
| M2 B | Linear SVM, short prompts with personal data (data fix), balanced own test prompts (600) | done |
| M2 C | Encoder model (XLM-RoBERTa + LoRA), two runs | done |
| M2 C | Small LLM (Qwen3-0.6B) + LoRA: notebook ready, run pending | in progress |
| M3 – M5 | API, final evaluation on the test set | planned |

**Results on the ai4privacy validation set** (31,380 texts, all languages):

| Model | Notebook | Macro F1 | Recall block | Recall block (de) | Latency p95 |
|---|---|---|---|---|---|
| Regex baseline | [`baseline_regex.ipynb`](notebooks/baseline_regex.ipynb) | 0.222 | 0.126 | 0.131 | 0.08 ms |
| TF-IDF + LogReg baseline | [`baseline_tfidf_logreg.ipynb`](notebooks/baseline_tfidf_logreg.ipynb) | 0.870 | 0.918 | 0.912 | 1.3 ms |
| TF-IDF + tuned (C = 30, block threshold 0.2) | [`tfidf_tuning.ipynb`](notebooks/tfidf_tuning.ipynb) | 0.864 | **0.955** | 0.953 | 2.9 ms |
| TF-IDF + harmless prompts in training | [`tfidf_harmless.ipynb`](notebooks/tfidf_harmless.ipynb) | 0.844 | **0.953** | 0.944 | 2.8 ms |
| Linear SVM tuned (C = 1.0, threshold -0.4) | [`svm_linear.ipynb`](notebooks/svm_linear.ipynb) | 0.865 | **0.956** | 0.955 | 3.4 ms |
| Linear SVM + harmless prompts | [`svm_linear.ipynb`](notebooks/svm_linear.ipynb) | 0.862 | **0.957** | 0.956 | 3.5 ms |
| TF-IDF + harmless + short prompts | [`short_prompts_retrain.ipynb`](notebooks/short_prompts_retrain.ipynb) | 0.852 | **0.954** | **0.959** | 4.0 ms ¹ |
| Linear SVM + harmless + short prompts | [`short_prompts_retrain.ipynb`](notebooks/short_prompts_retrain.ipynb) | 0.862 | **0.956** | **0.953** | 4.1 ms ¹ |
| Encoder XLM-R + LoRA + harmless (run 1) | [`train_encoder_lora_colab.ipynb`](notebooks/train_encoder_lora_colab.ipynb) | **0.963** | **0.994** | **0.994** | ~85–110 ms ² |
| Encoder XLM-R + LoRA + harmless + short prompts (run 2) | [`train_encoder_lora_colab.ipynb`](notebooks/train_encoder_lora_colab.ipynb) | **0.966** | **0.993** | **0.993** | not measured ³ |

¹ Measured in `short_prompts_retrain.ipynb` on another laptop; the "+ harmless" models measured there need
3.9 ms, so the short prompts do not make the models slower.
² CPU, one prompt at a time, on the same laptop as the SVM (3.5 ms), with `src/measure_latency.py`; two
measurements gave p95 = 85 ms and 109 ms. ³ Same model size as run 1, so about the same time.

**Results on our own German test prompts** (`data/own_prompts.csv`, 600 prompts, 200 per decision):

| Model | correct | warn → allow | block caught | harmless blocked |
|---|---|---|---|---|
| TF-IDF + harmless prompts | 449 of 600 | 66 of 200 | 185 of 200 | 20 of 200 |
| TF-IDF + harmless + short prompts | 484 of 600 | 34 of 200 | 189 of 200 | 19 of 200 |
| Linear SVM + harmless prompts | 455 of 600 | 51 of 200 | 190 of 200 | 29 of 200 |
| Linear SVM + harmless + short prompts | 486 of 600 | 20 of 200 | **191 of 200** | 28 of 200 |
| Encoder run 1 (+ harmless) | **548 of 600** | 6 of 200 | 177 of 200 | **7 of 200** |
| Encoder run 2 (+ harmless + short prompts) | **548 of 600** | **5 of 200** | 179 of 200 | 10 of 200 |

What we learned:

- **Harmless prompts** raised the share of correctly allowed real harmless prompts from 41 % to 93 %, and
  accuracy on our own prompts (v1, 200) from 65 % to 78 %. The downside: the TF-IDF / SVM models learned the
  shortcut "short request = allow" and let 15 to 20 of 60 warn prompts (names, e-mails) through.
- **Short prompts with personal data** (data fix, `src/short_prompts.py`) break this shortcut for TF-IDF and SVM:
  warn prompts let through fell from 66 to 34 (TF-IDF) and from 51 to 20 of 200 (SVM), without losing recall
  block on ai4privacy.
- **The encoder reads the context** instead of pieces of letters: best macro F1 on validation (0.96) and on our
  own prompts (548 of 600), only 5 to 6 of 200 warn prompts let through, and far fewer false alarms. The short
  prompts hardly change it (run 1 ≈ run 2): the encoder never had the shortcut problem.
- **Weaknesses of the encoder:** passwords written in everyday style ("Das WLAN-Passwort ist Kaffeebohne#42")
  are classified as warn instead of block (18 of the 21 missed block prompts of run 2), and order / ticket
  numbers cause most false alarms. It is 25 to 30 times slower than the SVM, with p95 around our 100 ms target.

### Next: a small LLM (Qwen3) with LoRA

Question: does the bigger language knowledge of an LLM give better results, above all a higher **block recall on our
own prompts** (encoder run 2: 179 of 200), and is it worth the loss in speed? Notebook:
[`train_llm_lora_colab.ipynb`](notebooks/train_llm_lora_colab.ipynb), explanation:
[`docs/algorithms/en/05_llm_lora.md`](docs/algorithms/en/05_llm_lora.md).

- Model: `Qwen/Qwen3-0.6B` (fixed revision, Apache-2.0) as a classifier with 3 classes, LoRA on
  `q_proj`/`k_proj`/`v_proj`/`o_proj` (4.6 M of 600 M weights trainable). Optional `Qwen3-1.7B` with QLoRA (4 bit).
- **Fair comparison:** the same data, sample, fingerprints, threshold rule and evaluation as encoder run 2 (shared code
  in `src/train_utils.py`, checked: identical training sample of 45,845 texts). Only the model is different.
- MLflow run `qwen3_0.6b_lora_templates`, comparison table `model_comparison_llm.csv` (SVM + short | encoder run 2 |
  Qwen3). Results will be added here after the run.
- First check of the speed (untrained adapter, laptop CPU): about **850 ms p95** per prompt, the encoder about 145 ms in the
  same test. So the LLM is clearly above our 100 ms target; the run shows whether the quality is worth it.

Details: [`reports/`](reports/), the conclusion section of each notebook, the runs in MLflow, and the
explanations of every model in [`docs/algorithms/`](docs/algorithms/) (German and English).

## Repository structure

```text
AIOps_PromptGuard/
├── app/                    FastAPI service (planned)
├── config/
│   └── policy.yaml         PII label → allow / warn / block
├── data/                   datasets (only splits.csv and own_prompts.csv are in Git)
│   ├── splits.csv          fixed train / validation / test split
│   └── own_prompts.csv     600 hand-written German test prompts (200 per label, column version v1/v2)
├── docs/algorithms/        how each model works, in simple words (de/ and en/):
│                           TF-IDF, Logistic Regression, linear SVM, encoder with LoRA, LLM with LoRA
├── models/                 trained models, not in Git (*.joblib, encoder adapter folders)
├── notebooks/              data preparation and experiments
│   ├── train_encoder_lora_colab.ipynb   encoder training, runs in Google Colab (GPU)
│   └── train_llm_lora_colab.ipynb       LLM (Qwen3) training, runs in Google Colab (GPU)
├── reports/                result tables and data checks
├── scripts/                one-off analysis scripts (e.g. check_harmless.py)
├── src/                    shared Python code
│   ├── data_utils.py       load splits, policy, fingerprints
│   ├── download_data.py    download ai4privacy dataset
│   ├── harmless_data.py    build harmless prompts (oasst1 + templates)
│   ├── short_prompts.py    build short prompts with and without personal data (all three classes)
│   ├── own_prompts.py      extend own_prompts.csv from 200 (v1) to 600 prompts (v2)
│   ├── make_splits.py      create / verify the split
│   ├── encoder_utils.py    load an encoder or LLM (base model + LoRA adapter) and classify prompts
│   ├── train_utils.py      training data, sample and evaluation shared by the encoder and LLM notebooks
│   ├── measure_latency.py  response time of an encoder or LLM on the CPU (laptop)
│   ├── evaluate_own_prompts.py  evaluate saved models from MLflow on own prompts v1 / v2 / all
│   └── check_mlflow.py     test the connection to the MLflow server
├── tests/                  unit tests (planned)
├── .env.example            template for the MLflow credentials
├── .python-version         Python version of the project (3.14)
├── requirements.txt        Python packages (incl. torch, transformers, peft for the encoder)
├── requirements-train.txt  extra packages only for GPU training in Colab (bitsandbytes for QLoRA)
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

The packages include PyTorch for the encoder model (about 200 MB). On Windows and macOS this is the CPU
version, which is all you need on a laptop. On Linux pip would download the much larger GPU version; add
`--extra-index-url https://download.pytorch.org/whl/cpu` to get the CPU version.

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
6. `svm_linear.ipynb` (compares with the models from steps 4 and 5)
7. `short_prompts_retrain.ipynb` (needs the "+ harmless" models from steps 5 and 6)
8. `train_encoder_lora_colab.ipynb`: needs a GPU, run it in Google Colab (step 10)
9. `train_llm_lora_colab.ipynb`: needs a GPU, run it in Google Colab (step 10)

### 10. Train the encoder in Google Colab

The encoder (XLM-RoBERTa + LoRA) needs a GPU. The notebook downloads the project from GitHub (branch `main`),
builds the data with the same scripts as step 7, checks the fingerprints, trains, evaluates and logs
everything to MLflow. How the model works: [`docs/algorithms/en/04_encoder_lora.md`](docs/algorithms/en/04_encoder_lora.md).

1. Push your changes to `main` (Colab clones the GitHub version, not your local files).
2. Open the notebook in Colab (badge at the top of the notebook) and choose
   *Runtime → Change runtime type → T4 GPU*.
3. Add two Colab secrets (key icon on the left, switch on "Notebook access"):
   `MLFLOW_TRACKING_USERNAME` (DagsHub user name) and `MLFLOW_TRACKING_PASSWORD` (DagsHub token).
4. In chapter 3 choose the run with `USE_TEMPLATE_PII`: `False` = run 1 (`xlmr_lora_benign`),
   `True` = run 2 with the short prompts (`xlmr_lora_benign_templates`).
5. *Runtime → Run all* (about 30 to 45 minutes). At the end the browser downloads a ZIP with the model
   (adapter) and the result tables; the model is also stored in the MLflow run under `model/`.

Afterwards, on the laptop (environment active):

```bash
# response time on the CPU; downloads the model from MLflow into models/ and logs latency_p50_ms / latency_p95_ms
python src/measure_latency.py --download --log                      # run 1
python src/measure_latency.py --download --log --run-name xlmr_lora_benign_templates --model-dir models/xlmr_lora_benign_templates

# saved models (SVM and encoder) on our own prompts v1 / v2 / all; --log adds own_prompts_v1_* / _v2_* / _all_*
python src/evaluate_own_prompts.py --log
```

#### The LLM (Qwen3 + LoRA)

`notebooks/train_llm_lora_colab.ipynb` works the same way (same Colab secrets, same steps 1 to 3). In chapter 3:

- `QUICK_TEST = True` (default): a mini run (about 1,000 texts, 1 epoch) to check that everything runs, about 10 to 15
  minutes. It is logged as `..._quicktest` with the tag `quick_test=true`. Then set `QUICK_TEST = False` and run all
  again (about 1 to 1.5 hours on a T4).
- `MODEL_SIZE = "0.6B"` (default, LoRA) or `"1.7B"` (QLoRA in 4 bits, needs `requirements-train.txt`, several hours).

Afterwards, on the laptop:

```bash
python src/measure_latency.py --download --log --run-name qwen3_0.6b_lora_templates
```

### 11. End your session

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
| `data/own_prompts.csv` (in Git) | 600 hand-written realistic German prompts, 200 per decision. Column `version`: v1 = the first 200 (80 allow, 60 warn, 60 block), v2 = 400 added to balance the classes (120 allow, 140 warn, 140 block, built with `src/own_prompts.py`). Column `style` = formal / casual | **evaluation only, never train or tune on them** |

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
from short_prompts import load_short_prompts, short_prompts_fingerprint

splits = load_all_splits()               # dict with "train", "validation", "test"
train, val = splits["train"], splits["validation"]
harmless_train = load_harmless("train")
short_train = load_short_prompts("train")
print(split_fingerprint(), harmless_fingerprint(), short_prompts_fingerprint())
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
4. `data/own_prompts.csv` is an extra test set: only look at the results, never tune on them. Notebooks run
   before the v2 extension report results on v1 only (200 prompts, 80/60/60); filter `version == "v1"` to
   reproduce them. In MLflow, use `own_prompts_v1_*`, `own_prompts_v2_*` and `own_prompts_all_*` to compare
   models (see [Experiment tracking](#experiment-tracking-with-mlflow)).
5. Log `split_fingerprint()`, `harmless_fingerprint()`, `short_prompts_fingerprint()` (if used) and the
   policy version with every MLflow run.
6. The test part of the short prompts (`split == "test"`) follows rule 3: only once, in Milestone 5.
7. Report results overall **and** for `language == "de"`.
8. Use Python 3.14 and `requirements.txt`, so results are reproducible across the team.

## Experiment tracking with MLflow

- All notebooks log to the shared DagsHub server when `.env` is set up (see
  [step 8](#8-connect-to-the-shared-mlflow-server-dagshub)), otherwise to a local `mlflow.db`.
- All runs belong to the experiment **`promptguard`**. Each run logs parameters, validation metrics and
  the fingerprints of the data it used.
- View the runs in DagsHub under **Experiments**, or locally with
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- The trained models are stored as artifacts of their run: `*.joblib` for TF-IDF / SVM, the folder `model/`
  (LoRA adapter, tokenizer, `promptguard_config.json`) for the encoder runs `xlmr_lora_benign` and
  `xlmr_lora_benign_templates` and the LLM run `qwen3_0.6b_lora_templates`.
- Quick tests of the LLM notebook have the tag `quick_test = true` (and `candidate = false`). Hide them in DagsHub
  with the filter `tags.quick_test != "true"`.
- **Metrics on our own prompts:** `own_prompts_*` was measured on v1 (200 prompts) in most runs, but on all
  600 prompts in runs logged after the v2 extension (the newest `svm_short` and `tfidf_logreg_short`).
  To compare models, use the explicit names `own_prompts_v1_*`, `own_prompts_v2_*` and `own_prompts_all_*`
  (added to the SVM and encoder runs with `src/evaluate_own_prompts.py`).

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
| `short_prompts_retrain.ipynb` stops because a model is missing | Run `tfidf_harmless.ipynb` and `svm_linear.ipynb` first. |
| Encoder: "LOAD REPORT … classifier … MISSING" when loading | Expected. The base model has no classifier; the trained one is loaded from the adapter right after. |
| Colab trains with an old version of the code | Colab keeps the first clone. *Runtime → Disconnect and delete runtime*, then run again; chapter 1 prints the git commit. |
| Colab: fingerprint of `harmless_prompts.csv` or `short_prompts.csv` is different | Upload your local file (folder icon on the left) into `data/` of the project and run the cell again. |
| LLM notebook: the training loss shows `nan` | The calculation in half precision overflowed. In chapter 8 set `fp16=False` (slower) and run again. |
| LLM notebook: "CUDA out of memory" | In chapter 3 halve `batch_size` and double `grad_accum` in `QWEN_MODELS` (the effective batch stays 32). |

## License and data sources

The project's own code and data (e.g. `src/`, `scripts/`, `notebooks/`, `app/`, `config/policy.yaml`,
`data/own_prompts.csv`, `data/splits.csv`) are licensed under the [MIT License](LICENSE).

Third-party data is **not** part of this repository (`data/*` is in `.gitignore`). It is downloaded by
our scripts and keeps its own license:

| Dataset | Used for | License | Downloaded by |
|---|---|---|---|
| [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | main training / validation / test data | Custom Ai4Privacy license ([license.md](https://huggingface.co/datasets/ai4privacy/pii-masking-200k/blob/main/license.md)): free for individuals, organisations with up to 3 staff and non-profits; derivatives must not be distributed, sold, re-licensed or sublicensed. Copyright © 2023 Ai4Privacy | `src/download_data.py` |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | harmless prompts | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | `src/harmless_data.py` |
| [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (model, not data) | base model of the encoder | MIT | the encoder notebook and `src/encoder_utils.py` |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (models, not data) | base model of the LLM classifier | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | the LLM notebook and `src/encoder_utils.py` |

- oasst1 citation: Köpf et al. (2023), "OpenAssistant Conversations: Democratizing Large Language Model
  Alignment", arXiv:2304.07327. We only use the first user message of each conversation and filter it
  (see `src/harmless_data.py`).
- `data/splits.csv` only contains row ids and split names, no texts from ai4privacy.
- Models trained on ai4privacy (e.g. `models/*.joblib`, the encoder adapters, MLflow artifacts) are covered by the ai4privacy
  terms, not by our MIT license. Do not publish them without checking those terms.
- A productive use of PromptGuard in a company with more than three employees would need a commercial
  Ai4Privacy license or retraining on freely licensed data.
