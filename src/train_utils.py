"""Training data and evaluation for the transformer notebooks (encoder and LLM), in one place.

WHY THIS FILE EXISTS
--------------------
The LLM notebook (notebooks/train_llm_lora_colab.ipynb) must use EXACTLY the same data and
the same evaluation as encoder run 2 ("xlmr_lora_benign_templates"), so that only the model
is different. Instead of copying the code a second time, it is here.

The code is taken unchanged from notebooks/train_encoder_lora_colab.ipynb:
    chapter 4   load_train_val()        train / validation table with the column "origin"
    chapter 5   sample_train()          2,500 ai4privacy texts per class x language + harmless + short prompts
    chapter 6   evaluate(), harmless_shares(), threshold_sweep(), choose_threshold(), plot_sweep()
    chapter 10  own_metrics()           the own_prompts_* metrics
    chapter 10b template_rows()         the "short val: ..." rows
The encoder notebook itself was not changed (it keeps its own copy), so its runs stay reproducible.

New here (only used by the LLM notebook):
    own_rows()               the "own: ..." rows of the comparison table, on all 600 own prompts
    own_rows_from_metrics()  the same rows, calculated from the metrics logged in another MLflow run

    from train_utils import load_train_val, sample_train, evaluate, threshold_sweep, choose_threshold
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from data_utils import load_all_splits
from encoder_utils import CLASSES, predict_with_threshold
from harmless_data import load_harmless

# The same targets and the same probability thresholds as in the TF-IDF and encoder notebooks
TARGET_RECALL_BLOCK = 0.95       # at least 95 % of block texts must be blocked
TARGET_FALSE_BLOCK = 0.05        # at most 5 % of allow texts may be blocked
THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

COLUMNS = ["source_text", "decision", "language"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------ data (encoder notebook, chapters 4 and 5)
def _tag(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    # keep only the columns we need and add where the text comes from
    return df[COLUMNS].assign(origin=origin)


def load_train_val(use_short_prompts: bool = True):
    """Train and validation table, like encoder notebook chapter 4. NO test set is loaded.

    Returns (train, val, short_val):
        train      ai4privacy train + harmless train (+ short prompts train), column "origin"
        val        ai4privacy validation + harmless validation (in this order)
        short_val  validation part of the short prompts (other sentence frames), ONLY for evaluation;
                   it is NOT added to val, so early stopping and the threshold use the same data as run 1.
                   None if use_short_prompts is False.
    """
    splits = load_all_splits()
    pii_train, pii_val = splits["train"], splits["validation"]
    del splits                                   # no test set

    train_parts = [_tag(pii_train, "ai4privacy"), _tag(load_harmless("train"), "harmless")]
    short_val = None
    if use_short_prompts:
        from short_prompts import load_short_prompts
        train_parts.append(_tag(load_short_prompts("train"), "template_pii"))
        short_val = _tag(load_short_prompts("validation"), "template_pii")
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat([_tag(pii_val, "ai4privacy"), _tag(load_harmless("validation"), "harmless")],
                    ignore_index=True)
    return train, val, short_val


def sample_train(train: pd.DataFrame, n_per_class_and_language: int = 2500, seed: int = 42,
                 use_short_prompts: bool = True) -> pd.DataFrame:
    """Training sample, like encoder notebook chapter 5.

    ai4privacy: shuffle once (seed), then the first n texts of every class x language group.
    Plus ALL harmless train prompts and (run 2) ALL short train prompts. Then all rows mixed.
    """
    pii_train = train[train["origin"] == "ai4privacy"]
    pii_sample = (pii_train.sample(frac=1, random_state=seed)
                  .groupby(["decision", "language"]).head(n_per_class_and_language))
    parts = [pii_sample, train[train["origin"] == "harmless"]]
    if use_short_prompts:
        parts.append(train[train["origin"] == "template_pii"])
    return (pd.concat(parts)
            .sample(frac=1, random_state=seed)          # mix the rows of all sources
            .reset_index(drop=True))


# ------------------------------------------------------------------ evaluation (encoder notebook, chapter 6)
def evaluate(y_true, y_pred) -> dict:
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


def harmless_shares(y_pred) -> dict:
    y_pred = np.asarray(y_pred)
    return {f"harmless -> {c}": float((y_pred == c).mean()) for c in CLASSES}


def threshold_sweep(scores, classes, y_true, rows_mask):
    # tries every threshold on the given rows and returns one table row per threshold
    rows = []
    for t in THRESHOLDS:
        pred = predict_with_threshold(scores[rows_mask], classes, t)
        rows.append({"threshold": t, **evaluate(np.asarray(y_true)[rows_mask], pred)})
    sweep = pd.DataFrame(rows)
    sweep["meets both targets"] = ((sweep["recall_block"] >= TARGET_RECALL_BLOCK)
                                   & (sweep["false_block_allow"] <= TARGET_FALSE_BLOCK))
    return sweep


def choose_threshold(sweep):
    # The same rule as in the TF-IDF notebooks, written down before looking at the results:
    # 1. highest threshold that reaches both targets, otherwise
    # 2. best recall block among thresholds with <= 5 % harmless texts blocked
    both = sweep[sweep["meets both targets"]]
    if len(both):
        return float(both["threshold"].max()), "highest threshold that reaches both targets"
    safe = sweep[sweep["false_block_allow"] <= TARGET_FALSE_BLOCK]
    if len(safe):
        return float(safe.loc[safe["recall_block"].idxmax(), "threshold"]), \
            "both targets not reachable; best recall block with <= 5 % false alarms"
    return None, "no threshold keeps false alarms <= 5 %; keep the default rule"


def plot_sweep(sweep, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sweep["threshold"], sweep["recall_block"], marker="o", color=BLUE, label="recall block (higher = safer)")
    ax.plot(sweep["threshold"], sweep["false_block_allow"], marker="o", color=ORANGE, label="allow texts blocked (lower = better)")
    ax.axhline(TARGET_RECALL_BLOCK, color=BLUE, linestyle="--", linewidth=1)
    ax.axhline(TARGET_FALSE_BLOCK, color=ORANGE, linestyle="--", linewidth=1)
    ax.invert_xaxis()
    ax.set_xlabel("block threshold t  (prompt is blocked if p(block) >= t)")
    ax.set_ylabel("share")
    ax.set_title(title)
    ax.legend(loc="center right")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ own prompts and short prompts (chapters 10, 10b)
def own_metrics(expected, pred, casual) -> dict:
    """The own_prompts_* metrics (same definitions as in all earlier runs)."""
    expected, pred, casual = np.asarray(expected), np.asarray(pred), np.asarray(casual)
    m = evaluate(expected, pred)
    m["accuracy"] = float((pred == expected).mean())
    m["warn_to_allow"] = float((pred[expected == "warn"] == "allow").mean())
    m["casual_recall_block"] = float((pred[(expected == "block") & casual] == "block").mean())
    return m


def template_rows(pred, y) -> dict:
    """The same four rows as in short_prompts_retrain.ipynb ("short val: ...")."""
    pred, y = np.asarray(pred), np.asarray(y)
    return {
        "short val: correct": f"{(pred == y).mean():.1%}",
        "short val: block caught": f"{(pred[y == 'block'] == 'block').mean():.1%}",
        "short val: warn let through (allow)": f"{(pred[y == 'warn'] == 'allow').mean():.1%}",
        "short val: allow blocked": f"{(pred[y == 'allow'] == 'block').mean():.1%}",
    }


def _count(hit: int, total: int) -> str:
    return f"{int(hit)} of {int(total)}"


def own_rows(expected, pred, casual) -> dict:
    """The "own ..." rows of the comparison tables as counts (same row names as
    reports/short_prompts_before_after.csv, which is also measured on all 600 prompts)."""
    expected, pred, casual = np.asarray(expected), np.asarray(pred), np.asarray(casual)
    block, warn, allow = expected == "block", expected == "warn", expected == "allow"
    return {
        "own prompts correct": _count((pred == expected).sum(), len(expected)),
        "own: block prompts caught": _count((pred[block] == "block").sum(), block.sum()),
        "own: casual block prompts caught": _count((pred[block & casual] == "block").sum(), (block & casual).sum()),
        "own: harmless prompts blocked": _count((pred[allow] == "block").sum(), allow.sum()),
        "own: warn prompts let through (allow)": _count((pred[warn] == "allow").sum(), warn.sum()),
    }


def own_rows_from_metrics(metrics: dict, prefix: str, expected, casual) -> dict:
    """The same rows as own_rows(), calculated from metrics logged in an MLflow run.

    Example: prefix "own_prompts_all_" and recall_block = 0.895 on 200 block prompts -> "179 of 200".
    Nothing is estimated: every logged share is a count divided by a known total.
    """
    expected, casual = np.asarray(expected), np.asarray(casual)
    block, warn, allow = expected == "block", expected == "warn", expected == "allow"

    def from_share(name: str, total: int) -> str:
        return _count(round(metrics[prefix + name] * total), total)

    return {
        "own prompts correct": from_share("accuracy", len(expected)),
        "own: block prompts caught": from_share("recall_block", block.sum()),
        "own: casual block prompts caught": from_share("casual_recall_block", (block & casual).sum()),
        "own: harmless prompts blocked": from_share("false_block_allow", allow.sum()),
        "own: warn prompts let through (allow)": from_share("warn_to_allow", warn.sum()),
    }
