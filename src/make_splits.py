"""Create the fixed train / validation / test split for all PromptGuard models.

Split rules:
    - 70 % train / 15 % validation / 15 % test
    - stratified by language x policy decision (allow / warn / block)
    - grouped by masked template (`target_text`): all texts that share the same
      template end up in the same split (no template leakage into test)
    - lorem-ipsum filler texts are excluded
    - fixed seed (42)

Output: data/splits.csv with the columns `id` and `split`
        (split = train / validation / test / excluded).

Usage (from the project folder, with the venv activated):
    python src/make_splits.py            # creates data/splits.csv
    python src/make_splits.py --force    # overwrites an existing data/splits.csv

The split is created ONCE and then reused by every model. Do not re-create it
during the project, otherwise results of different models are not comparable.

--------
stratified : every group (here: language x decision, e.g. "de_allow") has the
             same share in train, validation and test. Without this, the small
             group "allow" (~8 %) could by chance be rarer in the test set.
grouped    : the dataset is synthetic - many texts were generated from the same
             sentence template, only with different names / numbers filled in.
             The masked text (`target_text`, e.g. "Hello [FIRSTNAME], ...") is
             that template. If one variant is in train and another in test, the
             model has "seen" the test sentence already and the test score is
             too optimistic. So all variants of a template stay in ONE split.
seed       : start value of the random number generator. The same seed always
             produces the same "random" order, so the split is reproducible.
"""

from __future__ import annotations

import argparse          # reads command line options like --force
import sys

import numpy as np
import pandas as pd

# Our own helper functions (src/data_utils.py)
from data_utils import (
    SPLITS_PATH,
    add_decision,
    is_filler_text,
    load_raw,
    split_fingerprint,
)

# --------------------------------------------------------------------------- settings
SEED = 42
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15  # the remaining 15 % are test


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Return a Series (same index as df) with "train" / "validation" / "test".

    How it works:
      1. Every template (`target_text`) is one group. Each group gets a stratum
         (language_decision) - texts of one template share the same PII labels,
         so they also share the same decision.
      2. Inside each stratum the groups are shuffled with the fixed seed.
      3. Walking through the shuffled groups, the first 70 % of the texts go to
         train, the next 15 % to validation and the rest to test.
    Result: exact 70/15/15 per stratum, and no template is split across sets.
    """
    # Stratum = the group we stratify by, e.g. "de_allow" or "fr_block"
    df = df.assign(stratum=df["language"] + "_" + df["decision"])

    # --- Step 1: one row per template ---------------------------------------
    # groupby("target_text") collects all texts with the same template.
    # For each template we store:
    #   stratum = the most frequent stratum of its texts (.mode() = most frequent value)
    #   n       = how many texts belong to this template (usually 1, at most 7)
    # groupby sorts the templates alphabetically -> the order is always the same,
    # which is required so that the seed produces the same result every time.
    groups = df.groupby("target_text").agg(
        stratum=("stratum", lambda s: s.mode().iloc[0]), n=("stratum", "size")
    )

    # --- Step 2: shuffle the templates (reproducibly) ------------------------
    # Give every template a random number between 0 and 1 (always the same
    # numbers thanks to the fixed seed) and sort by stratum, then by that number.
    # Result: within each stratum the templates are in a random order.
    rng = np.random.default_rng(SEED)
    groups["random_key"] = rng.random(len(groups))
    groups = groups.sort_values(["stratum", "random_key"])

    # --- Step 3: cut each stratum at 70 % and 85 % -----------------------------
    # cum   = running total of texts within the stratum (e.g. 1, 2, 4, 5, ...)
    # total = number of texts in the whole stratum
    # position = where the template lies within its stratum, from 0 (start) to 1 (end).
    # We use the template's centre (cum - n/2), so a template with several texts
    # is assigned to the split in which most of it would fall.
    cum = groups.groupby("stratum")["n"].cumsum()
    total = groups.groupby("stratum")["n"].transform("sum")
    position = (cum - groups["n"] / 2) / total

    # np.select checks the conditions in order and takes the first one that is true:
    #   position < 0.70 -> train,  position < 0.85 -> validation,  otherwise -> test
    groups["split"] = np.select(
        [position < TRAIN_SHARE, position < TRAIN_SHARE + VALIDATION_SHARE],
        ["train", "validation"],
        default="test",
    )

    # --- Step 4: give every text the split of its template ---------------------
    # .map() looks up each text's template in the groups table
    return df["target_text"].map(groups["split"])


def print_report(df: pd.DataFrame) -> None:
    """Print checks so we can see that the split is correct."""
    used = df[df["split"] != "excluded"]

    # Check 1: are the split sizes 70 / 15 / 15 %?
    print("\nSplit sizes:")
    sizes = used["split"].value_counts().reindex(["train", "validation", "test"])
    print(pd.DataFrame({"texts": sizes, "%": (sizes / sizes.sum() * 100).round(1)}))
    print(f"excluded (filler texts): {(df['split'] == 'excluded').sum()}")

    # Check 2: does every language x decision group have the same share in all splits?
    # crosstab counts the texts per (language, decision) and split;
    # normalize="columns" turns the counts into shares per split (each column sums to 1).
    print("\nShare of language x decision per split (%):")
    table = pd.crosstab(
        [used["language"], used["decision"]], used["split"], normalize="columns"
    ) * 100
    print(table[["train", "validation", "test"]].round(2))

    # Check 3: is every template in exactly one split?
    # nunique() counts in how many different splits a template appears;
    # .gt(1) marks templates that appear in more than one split, .sum() counts them.
    leaks = used.groupby("target_text")["split"].nunique().gt(1).sum()
    print(f"\nTemplates that appear in more than one split: {leaks}")
    if leaks:
        raise RuntimeError("Template leakage between splits - check the split logic.")


def main() -> None:
    # --- Read command line options ---------------------------------------------
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="overwrite data/splits.csv")
    args = parser.parse_args()

    # --- Protect the existing split ----------------------------------------------
    # If the split already exists we do NOT overwrite it by accident.
    if SPLITS_PATH.exists() and not args.force:
        print(f"{SPLITS_PATH} already exists (fingerprint {split_fingerprint()}).")
        print("Use --force only if the team agreed to create a new split.")
        sys.exit(0)

    # --- Load data and calculate the decision per text -----------------------------
    print("Loading data ...")
    # We only need these columns for the split -> saves memory
    df = load_raw()[["id", "language", "target_text", "source_text", "privacy_mask"]]
    df = add_decision(df)

    # --- Mark filler texts as "excluded", split all other texts --------------------
    filler = df["source_text"].apply(is_filler_text)   # True/False per text
    df["split"] = "excluded"
    print("Assigning splits ...")
    # ~filler means "NOT filler"; only these texts are split
    df.loc[~filler, "split"] = assign_splits(df[~filler])

    print_report(df)

    # --- Save -------------------------------------------------------------------------
    # Only id + split are stored (small file). Sorting by id makes the file stable.
    # lineterminator="\n" writes the same line breaks on Windows and macOS.
    df[["id", "split"]].sort_values("id").to_csv(SPLITS_PATH, index=False, lineterminator="\n")
    print(f"\nWrote {SPLITS_PATH} (fingerprint {split_fingerprint()})")


# This block only runs when the file is started directly (python src/make_splits.py),
# not when it is imported by another file.
if __name__ == "__main__":
    main()
