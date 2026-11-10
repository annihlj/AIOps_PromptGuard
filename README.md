# AIOps_PromptGuard
PromptGuard is a machine learning project that checks employee prompts for sensitive information before they are sent to an external AI service.

## Goal
Help IT security and compliance teams reduce accidental disclosure of personal data and credentials.

## How it works
The system detects sensitive text and applies a defined policy to return:

- **Allow:** No relevant sensitive information detected.
- **Warn:** Review or edit the prompt before sending.
- **Block:** Content violates the defined policy.

## Project approach
1. Prepare selected categories from datasets. (**ai4privacy/pii-masking-200k** ...)
2. Build a regex and TF-IDF Logistic Regression baseline for emails, phone numbers and bank details.
3. Train and evaluate a small Named Entity Recognition (NER) model. (And try other models for comparison) 
4. Track experiments with **MLflow**.
5. Expose the selected model through **FastAPI**.

## Evaluation
Compare detection quality, missed sensitive content, false alarms and response times.

## Tech stack
Python; Jupyter; Hugging Face; MLflow; FastAPI; GitHub

## Data, Policy and Split

### Dataset
[`ai4privacy/pii-masking-200k`](https://huggingface.co/datasets/ai4privacy/pii-masking-200k),
the German, English, French and Italian files (209,261 synthetic texts, 56 PII labels).
The first analysis is in [`notebooks/preperation.ipynb`](notebooks/preperation.ipynb).

### Policy: PII label → allow / warn / block
[`config/policy.yaml`](config/policy.yaml) maps every PII label to a decision
(current version is a proposal):

- **block:** credentials and financial / government identifiers (password, PIN, credit card, IBAN, account number, SSN, ...)
- **warn:** data that identifies or locates a person (names, e-mail, phone, address, date of birth, IP, ...)
- **allow:** context data that does not identify a person on its own (date, amount, job title, city, ...)

A text gets the **most severe** decision of all its PII entities (block > warn > allow).
Result on the full dataset: 7.8 % allow, 60.7 % warn, 31.4 % block.

### Train / validation / test split
All models use the **same fixed split**, stored in [`data/splits.csv`](data/splits.csv)
(columns `id`, `split`; committed to Git).

| Rule | Value |
|---|---|
| Shares | 70 % train / 15 % validation / 15 % test |
| Stratified by | language × policy decision (e.g. `de_allow`, `fr_block`) |
| Grouped by | masked template (`target_text`): texts that share a template are always in the same split, so the test set contains no templates seen in training |
| Excluded | 41 lorem-ipsum filler texts (`split = excluded`) |
| Seed | 42 |
| Fingerprint | `6afd6a257de1` (SHA-256 of `splits.csv`, first 12 characters) |

Resulting sizes: train 146,455 · validation 31,380 · test 31,385.
Every language × decision group has the same share in all three splits.

The split was created with `python src/make_splits.py`. The script is
deterministic (running it again gives the same fingerprint) and refuses to
overwrite an existing `splits.csv` unless `--force` is passed.
**Do not re-create the split** unless the whole team agrees. Otherwise results
of different models are no longer comparable.

> The split is fixed independently of the policy. If labels in `policy.yaml`
> are moved later, the `decision` column changes, but the split stays the same.

### Using the data for model training
Load the data **only** through [`src/data_utils.py`](src/data_utils.py).
From a notebook inside `notebooks/`:

```python
import sys
sys.path.append("../src")
from data_utils import load_all_splits, split_fingerprint

splits = load_all_splits()          # dict with "train", "validation", "test"
train = splits["train"]
val   = splits["validation"]
print("split fingerprint:", split_fingerprint())
```

Each split is a DataFrame with all original dataset columns plus `decision`:

| Column | Use it for |
|---|---|
| `source_text` | model input (the prompt) |
| `decision` | target for classifiers (allow / warn / block) |
| `privacy_mask` | character spans + labels, for regex / span-based evaluation |
| `mbert_text_tokens`, `mbert_bio_labels` | token-level BIO labels for NER models |
| `language` | per-language evaluation (especially `de`) |

Other helpers: `load_split("train")` loads a single split,
`load_policy()` / `prompt_decision(privacy_mask, policy)` turn detected PII labels
into a decision. Use the same function to turn NER predictions into allow / warn / block.

**Rules for every model**
1. Train only on `train`.
2. Tune hyperparameters, thresholds and compare models only on `validation`.
3. Evaluate on `test` **once**, at the end, for the final chosen models.
4. Log `split_fingerprint()` and the policy version with every MLflow run.
5. Report results overall **and** for `language == "de"`.
6. Realistic German prompts (written by us) are kept as a separate, extra test set
   and are not part of this split.

## Setting Up the Environment and Running the Notebooks

Our Jupyter notebooks (`.ipynb` files) use a virtual Python environment
(venv). This keeps the project's Python packages separate from other projects.

Each developer creates their own environment locally.
The instructions below support Windows and macOS.

### 1. Prerequisites

- Install Python: https://www.python.org/downloads/
- Use the Python version agreed upon for this project.
- Download or clone the project, including `requirements.txt`.
- Use either VS Code or JupyterLab to run the notebooks.

Check whether Python is installed:

**Windows:**

```powershell
py --version
```

If `py` is unavailable, try `python --version`.
In that case, also use `python` instead of `py` when creating the environment.

**macOS:**

```bash
python3 --version
```

### 2. Open a Terminal in the Project Folder

Open a terminal and navigate to the main project folder containing
`requirements.txt`:

```bash
cd "path/to/AIOps_PromptGuard"
```

Replace the example path with your actual project path.

### 3. Create the Virtual Environment

You only need to do this once when setting up the project.

**Windows:**

```powershell
py -m venv .venv
```

**macOS:**

```bash
python3 -m venv .venv
```

This creates a folder named `.venv` inside the project folder.

> This guide uses `.venv` as the environment folder name.
> If your existing environment is named `venv`, replace `.venv`
> with `venv` in the commands below.

### 4. Activate the Virtual Environment

Activate the environment in each new terminal session you use for this project.

**Windows — PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows — Command Prompt (CMD):**

```cmd
.venv\Scripts\activate.bat
```

**macOS — Terminal:**

```bash
source .venv/bin/activate
```

Once activated, `(.venv)` usually appears at the beginning of the terminal prompt.

Check which Python installation is being used:

```bash
python -c "import sys; print(sys.executable)"
```

The output should point to Python inside this project's `.venv` folder.

> From this point onward, Windows and macOS use the same `python`
> commands, provided the virtual environment is activated.

### 5. Install the Project Dependencies

With the virtual environment activated, run the following command
from the project folder:

```bash
python -m pip install -r requirements.txt
```

This installs the packages listed in `requirements.txt` into the environment.

If `requirements.txt` changes later, run the same command again.

### 5b. Download the Dataset

The data files are not stored in Git (they are too large). Download them
from Hugging Face (`ai4privacy/pii-masking-200k`) into the `data/` folder:

```bash
python src/download_data.py
```

Files that already exist are skipped.

##### Dataset Documentation

See the original
[dataset card](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
for details about the dataset, its structure, and its usage terms.

### 5c. Check the Train / Validation / Test Split

The split file `data/splits.csv` is part of the repository. Check that you have it:

```bash
python src/make_splits.py
```

The script only prints the existing fingerprint (`6afd6a257de1`) and does not
change anything. See [Train / validation / test split](#train--validation--test-split).

### 6. Register the Environment as a Notebook Kernel

A kernel is the Python process that executes a notebook's code cells.

With the virtual environment activated, install the kernel package:

```bash
python -m pip install ipykernel
```

Then register the environment:

```bash
python -m ipykernel install --user --name project-venv --display-name "Python (Project Venv)"
```

The environment will appear in the kernel selection menu as
**Python (Project Venv)**.

Registration is normally only required once.
Repeat the registration if you delete or recreate the environment,
or move the project folder.

> If you work on multiple projects, use a unique kernel name and display
> name for each project instead of `project-venv` and `Python (Project Venv)`.

### 7. Run the Notebooks

Choose one of the following options.

#### VS Code

1. Install Microsoft's **Python** and **Jupyter** extensions in VS Code.
2. Open the entire project folder in VS Code.
3. Open the `.ipynb` file you want to run.
4. Click **Select Kernel** in the top-right corner of the notebook.
5. Select **Python (Project Venv)**. If necessary, first select
   **Select Another Kernel → Jupyter Kernels**.
6. Alternatively, select **Python Environments** and choose this project's
   `.venv` directly.
7. Run the cells from top to bottom:
   - Click the play button next to a code cell.
   - Press `Shift + Enter` to run the current cell.
   - Click **Run All** to execute the entire notebook.

**Important:** Check the selected kernel for each notebook.
Activating the environment in the terminal does not automatically
select it as the notebook's kernel.

### 8. Finish Your Session and Deactivate the Environment

Save your notebooks and shut down their running kernels.

If you are using JupyterLab, stop the server by pressing `Ctrl + C`
in its terminal and confirm if prompted.

Then deactivate the virtual environment:

```bash
deactivate
```

> `deactivate` only deactivates the environment in the current terminal.
> It does not shut down notebook kernels that are already running.

### Team Collaboration Notes

- Each developer creates their own local virtual environment.
- Share notebooks and `requirements.txt` through Git.
- Do not commit `.venv` or copy it between computers.
- Use matching Python and package versions where possible.
  Any operating-system-specific dependencies must account for
  both Windows and macOS.
- Document required datasets and where they should be stored.
- Include `ipykernel` in the project's dependency list.
  Include `jupyterlab` too if browser-based execution is part of the setup.

Add these entries to `.gitignore` if they are not already present:

```gitignore
.venv/
.ipynb_checkpoints/
```
