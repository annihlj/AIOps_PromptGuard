Olga, Eren, Ilja
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
