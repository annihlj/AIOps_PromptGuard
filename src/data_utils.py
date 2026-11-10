"""Shared data helpers for all PromptGuard models.

WHY THIS FILE EXISTS
--------------------
We train several models (regex, TF-IDF + Logistic Regression, NER, ...).
To compare them fairly, every model must use
    - exactly the same train / validation / test split, and
    - exactly the same allow / warn / block labels.
So instead of every notebook loading the data in its own way, all notebooks
use the functions in this file.

TYPICAL USAGE (in a notebook inside the folder notebooks/)
----------------------------------------------------------
    import sys
    sys.path.append("../src")          # lets Python find this file
    from data_utils import load_all_splits, split_fingerprint

    splits = load_all_splits()         # dict with "train", "validation", "test"
    train = splits["train"]
    val = splits["validation"]
"""

# Allows modern type hints like "dict | None" in older Python versions
from __future__ import annotations

import hashlib          # to calculate a fingerprint (hash) of the split file
import re               # regular expressions, used to detect lorem-ipsum texts
from pathlib import Path

import pandas as pd
import yaml             # to read config/policy.yaml

# --------------------------------------------------------------------------- paths
# Path(__file__) is the path of THIS file (src/data_utils.py).
# .parent is the folder src/, .parent.parent is the project folder.
# Building paths this way means the code works no matter from where it is started.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
POLICY_PATH = PROJECT_DIR / "config" / "policy.yaml"
SPLITS_PATH = DATA_DIR / "splits.csv"

# The four language files of the dataset (same files as in download_data.py)
DATA_FILES = [
    "german_pii_52k.jsonl",
    "english_pii_43k.jsonl",
    "french_pii_62k.jsonl",
    "italian_pii_50k.jsonl",
]

# The only allowed split names
SPLIT_NAMES = ("train", "validation", "test")

# Some synthetic texts are just lorem-ipsum filler ("Donec tincidunt id tortor ...")
# with PII inserted. They do not look like real prompts, so we exclude them.
# This pattern matches typical filler words; \b means "word boundary", so e.g.
# "ipsum" only matches as a whole word. re.IGNORECASE ignores upper/lower case.
_FILLER_PATTERN = re.compile(
    r"\b(lorem|ipsum|dolor sit|consectetur|adipiscing|vestibulum|nullam|donec|aliquam|pellentesque)\b",
    re.IGNORECASE,
)


# =========================================================================== raw data
def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read all four language files into one DataFrame (no filtering, no split)."""
    frames = []
    for filename in DATA_FILES:
        path = data_dir / filename
        # Give a helpful message instead of a cryptic error if the data is missing
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run `python src/download_data.py` first."
            )
        # lines=True: the file is JSON Lines = one JSON object (one text) per line
        frames.append(pd.read_json(path, lines=True))
    # Stack the four tables on top of each other; ignore_index=True creates a new
    # continuous row index 0, 1, 2, ... instead of repeating the indices of each file
    return pd.concat(frames, ignore_index=True)


def is_filler_text(text: str) -> bool:
    """True for lorem-ipsum filler texts (at least two typical filler words)."""
    # findall returns a list of all matches; two or more matches = filler text.
    # (One single match could be a coincidence, e.g. a name "Donec".)
    return len(_FILLER_PATTERN.findall(text)) >= 2


# =========================================================================== policy
def load_policy(path: Path = POLICY_PATH) -> dict:
    """Read config/policy.yaml and prepare it for fast lookups.

    Returns a dict with:
        "label_to_decision": {"IBAN": "block", "EMAIL": "warn", "DATE": "allow", ...}
        "order":             ["block", "warn", "allow"]   (most severe first)
        "default":           "allow"                      (text without any PII)
    """
    with open(path, encoding="utf-8") as f:
        policy = yaml.safe_load(f)

    # In the YAML file the labels are grouped by decision:
    #     block: [PASSWORD, PIN, ...]
    #     warn:  [FIRSTNAME, ...]
    # Here we "turn it around" into one entry per label, so we can later look up
    # the decision of a label in one step: label_to_decision["IBAN"] -> "block"
    label_to_decision = {
        label: decision
        for decision, labels in policy["labels"].items()
        for label in labels
    }
    return {
        "label_to_decision": label_to_decision,
        "order": policy["decision_order"],      # most severe first
        "default": policy["default_decision"],
    }


def prompt_decision(privacy_mask: list[dict], policy: dict) -> str:
    """Decision for ONE text: the most severe decision of all its PII entities wins.

    privacy_mask is the list of PII entities of the text, e.g.
        [{"label": "FIRSTNAME", ...}, {"label": "IBAN", ...}]
    -> FIRSTNAME = warn, IBAN = block -> the text is "block".

    Labels that are missing in the policy are treated as "warn" (fail safe:
    better to warn too often than to let unknown data through).
    """
    # A set collects every decision only once, e.g. {"warn", "block"}
    decisions = {
        policy["label_to_decision"].get(entry["label"], "warn") for entry in privacy_mask
    }
    # Go through the decisions from most severe to least severe and return the
    # first one that occurs in this text
    for decision in policy["order"]:
        if decision in decisions:
            return decision
    # No PII entity at all -> default decision ("allow")
    return policy["default"]


def add_decision(df: pd.DataFrame, policy: dict | None = None) -> pd.DataFrame:
    """Return a copy of df with an extra column `decision` (allow / warn / block)."""
    # If no policy is given, read it from config/policy.yaml
    policy = policy or load_policy()
    # .copy() so that the DataFrame passed in by the caller is not changed
    out = df.copy()
    # .apply() calls prompt_decision once for every row's privacy_mask
    out["decision"] = out["privacy_mask"].apply(lambda m: prompt_decision(m, policy))
    return out


# =========================================================================== splits
def load_split_ids(path: Path = SPLITS_PATH) -> pd.DataFrame:
    """Read data/splits.csv: one row per text with the columns `id` and `split`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python src/make_splits.py` first."
        )
    return pd.read_csv(path)


def split_fingerprint(path: Path = SPLITS_PATH) -> str:
    """Short "fingerprint" (hash) of splits.csv.

    A hash is a short code calculated from the file content: the same content
    always gives the same code, and any change gives a different code.
    Log it (e.g. in MLflow) with every training run to prove that all models
    used the same split.

    Windows saves line breaks as "\\r\\n", macOS as "\\n". We replace one by the
    other before hashing, so both systems get the same fingerprint.
    """
    content = path.read_bytes().replace(b"\r\n", b"\n")
    # sha256 is a standard hash function; we only keep the first 12 characters,
    # which is short to read but still practically unique
    return hashlib.sha256(content).hexdigest()[:12]


def load_split(name: str, with_decision: bool = True) -> pd.DataFrame:
    """Load ONE split ("train", "validation" or "test") with all dataset columns.

    with_decision=True adds the column `decision` based on config/policy.yaml.
    Tip: if you need several splits, use load_all_splits() - it is faster
    because it reads the large raw files only once.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"name must be one of {SPLIT_NAMES}, got {name!r}")

    ids = load_split_ids()
    # All ids that belong to the wanted split; a set makes the lookup fast
    wanted = set(ids.loc[ids["split"] == name, "id"])

    df = load_raw()
    # Keep only the rows of this split; reset_index gives a clean index 0, 1, 2, ...
    df = df[df["id"].isin(wanted)].reset_index(drop=True)
    return add_decision(df) if with_decision else df


def load_all_splits(with_decision: bool = True) -> dict[str, pd.DataFrame]:
    """Load train, validation and test at once.

    Returns a dict: {"train": DataFrame, "validation": DataFrame, "test": DataFrame}
    Texts with split = "excluded" (lorem-ipsum filler) are not returned.
    """
    ids = load_split_ids()
    # .merge() adds the column `split` to every text by matching the `id` column.
    # how="inner" keeps only ids that exist in both tables.
    df = load_raw().merge(ids, on="id", how="inner")
    if with_decision:
        df = add_decision(df)

    # Build one DataFrame per split; the column `split` is no longer needed there
    return {
        name: df[df["split"] == name].drop(columns="split").reset_index(drop=True)
        for name in SPLIT_NAMES
    }
