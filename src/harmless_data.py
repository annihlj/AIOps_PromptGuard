"""Extra HARMLESS prompts (decision = allow) for PromptGuard.

WHY THIS FILE EXISTS
--------------------
Every text in ai4privacy/pii-masking-200k contains at least two PII entities.
So the model has never seen a truly harmless prompt such as
"Fasse den Text zusammen". The result: false alarms on normal business prompts
(e.g. "Erstelle eine Tabelle mit den Umsätzen 2023 ..." was predicted block).

This file builds a second, small dataset of prompts WITHOUT personal data,
all with the decision "allow". It comes from two sources:

1. oasst1 (OpenAssistant Conversations, Apache-2.0 licence)
   Real questions that real people typed into a chat assistant, in German,
   English, French and Italian. We only take the FIRST message of each
   conversation (what a user asks first) and drop messages that look like they
   contain personal data (see _looks_like_pii).
   -> split 70 / 15 / 15 into train / validation / test.

2. Company-style templates (written by our team, see TEMPLATES_DE / _EN)
   Typical office requests ("Erstelle eine Checkliste für ...") combined with
   harmless details (departments, dates, amounts, cities).
   -> ONLY used for training. They are generated, so measuring on them would be
   too easy. Validation and test only contain real human prompts (oasst1).

THE RULES (same spirit as data/splits.csv)
-------------------------------------------
- The split is DETERMINISTIC: it is calculated from a hash of the conversation
  id, so everybody who runs this script gets exactly the same split.
- Our own test prompts (data/own_prompts.csv) are never used here. The script
  removes any harmless prompt that is identical to one of them.
- The test part is only used once, at the end of the project (Milestone 5).

USAGE
-----
Build the file once (from the project folder, venv activated):
    python src/harmless_data.py

In a notebook:
    sys.path.append("../src")
    from harmless_data import load_harmless, harmless_fingerprint
    harmless_train = load_harmless("train")
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "harmless_prompts.csv"
OWN_PROMPTS_PATH = DATA_DIR / "own_prompts.csv"

# oasst1 files on Hugging Face (repo "OpenAssistant/oasst1", folder data/)
OASST_REPO = "OpenAssistant/oasst1"
OASST_FILES = [
    "data/train-00000-of-00001-b42a775f407cee45.parquet",
    "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
]
LANGUAGES = ["de", "en", "fr", "it"]      # the same four languages as our main dataset
MAX_PER_LANGUAGE = 3000                   # stops English from dominating (oasst1 is mostly English)
MIN_CHARS, MAX_CHARS = 15, 600            # very short or very long texts do not look like our prompts
N_TEMPLATE_PROMPTS = 2000                 # how many company-style prompts we generate for training
SEED = 42


# =========================================================================== PII filter
# We want prompts WITHOUT personal data. A regex cannot find every name, but it
# removes the obvious cases. Anything that matches one of these patterns is dropped.
_PII_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                       # e-mail address
    re.compile(r"@\w{3,}"),                                       # social media handle
    re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,}"),          # IBAN-like
    re.compile(r"(?:\+|\b00)\d[\d\s/-]{6,}\d"),                    # international phone number
    re.compile(r"\b\d[\d\s/-]{7,}\d\b"),                           # long digit sequence (phone, account, card)
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                    # IPv4 address
    # a password / PIN that is actually GIVEN ("password: abc", "PIN ist 1234"),
    # questions ABOUT passwords ("how do I make a strong password?") are kept
    re.compile(r"\b(?:pass(?:wort|word)?|pw|kennwort|pin|passcode)\s*(?:ist|is|lautet|:|=)\s*\S+", re.IGNORECASE),
]


def _looks_like_pii(text: str) -> bool:
    return any(p.search(text) for p in _PII_PATTERNS)


def _normalise(text: str) -> str:
    """Lowercase + single spaces, used to find duplicates."""
    return " ".join(text.lower().split())


def _bucket(key: str) -> str:
    """Deterministic 70 / 15 / 15 split from a hash of the key."""
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


# =========================================================================== source 1: oasst1
def _load_oasst() -> pd.DataFrame:
    """Download oasst1 (once) and return the first user message of every conversation."""
    frames = []
    for filename in OASST_FILES:
        local = DATA_DIR / "oasst1" / Path(filename).name
        if not local.exists():
            from huggingface_hub import hf_hub_download   # same library as src/download_data.py
            print(f"[download] {filename} ...")
            hf_hub_download(repo_id=OASST_REPO, filename=filename, repo_type="dataset",
                            local_dir=DATA_DIR / "oasst1")
            # hf_hub_download keeps the folder "data/" of the repo -> move the file up one level
            downloaded = DATA_DIR / "oasst1" / filename
            if downloaded.exists() and downloaded != local:
                downloaded.replace(local)
        frames.append(pd.read_parquet(local))
    df = pd.concat(frames, ignore_index=True)

    # role == "prompter" -> written by a human user (not the assistant)
    # parent_id is empty -> the FIRST message of a conversation (a fresh request)
    df = df[(df["role"] == "prompter") & (df["parent_id"].isna()) & (df["lang"].isin(LANGUAGES))]
    if "deleted" in df.columns:
        df = df[df["deleted"] != True]   # noqa: E712  (column may contain None)

    return pd.DataFrame({
        "source_text": df["text"].str.strip(),
        "language": df["lang"],
        "group": df["message_tree_id"],   # one conversation = one group
        "source": "oasst1",
    })


# =========================================================================== source 2: templates
# Typical office requests. {placeholders} are filled with harmless details below.
# These were written independently of data/own_prompts.csv (our test prompts).
TEMPLATES_DE = [
    "Fasse {doc} in {n} Stichpunkten zusammen.",
    "Schreib eine kurze Mail an {group} wegen {topic}.",
    "Erstelle eine Checkliste für {topic}.",
    "Formuliere {doc} etwas freundlicher.",
    "Übersetze {doc} ins Englische.",
    "Erkläre mir {concept} in einfachen Worten.",
    "Mach aus {doc} eine Präsentation mit {n} Folien.",
    "Welche Risiken gibt es bei {topic}?",
    "Gib mir {n} Ideen für {topic}.",
    "kannst du {doc} kürzer machen",
    "schreib ne kurze info an {group} wegen {topic}",
    "Erstelle eine Tabelle mit den Spalten {cols}.",
    "Plane {topic} für den {date}, Budget {amount}.",
    "Wie berechne ich {kpi} für {dept}?",
    "Formuliere eine Rundmail: {topic} findet am {date} in {city} statt.",
]
TEMPLATES_EN = [
    "Summarize {doc} in {n} bullet points.",
    "Write a short email to {group} about {topic}.",
    "Create a checklist for {topic}.",
    "Make {doc} sound more friendly.",
    "Explain {concept} in simple words.",
    "Turn {doc} into a presentation with {n} slides.",
    "What are the risks of {topic}?",
    "Give me {n} ideas for {topic}.",
    "can you make {doc} shorter",
    "Create a table with the columns {cols}.",
    "Plan {topic} for {date}, budget {amount}.",
    "How do I calculate {kpi} for {dept}?",
]
FILLERS = {
    "de": {
        "doc": ["diesen Text", "das Protokoll", "die Präsentation", "den Bericht", "diese Mail",
                "das Konzept", "die Projektbeschreibung", "den Absatz"],
        "group": ["das Vertriebsteam", "alle Mitarbeitenden", "die Abteilungsleitung", "das Lager",
                  "den Einkauf", "die IT-Abteilung", "das Projektteam"],
        "topic": ["die Inventur", "das Sommerfest", "die neue Reiserichtlinie", "die Umstellung auf SAP S/4HANA",
                  "die Messe im Frühjahr", "das Onboarding", "die Preisanpassung", "die Kundenumfrage",
                  "den Umzug ins neue Büro", "die Datenschutzschulung", "die Jahresplanung", "das Lieferantenaudit"],
        "concept": ["Cashflow", "Abschreibungen", "agiles Projektmanagement", "eine Balanced Scorecard",
                    "Machine Learning", "die DSGVO", "Lean Management", "Zwei-Faktor-Authentifizierung"],
        "kpi": ["den Deckungsbeitrag", "die Umsatzrendite", "die Lagerumschlagshäufigkeit", "den ROI"],
        "dept": ["den Vertrieb", "das Lager", "das Marketing", "die Produktion"],
        "cols": ["Monat, Umsatz, Kosten", "Abteilung, Budget, Ist-Wert", "Produkt, Menge, Preis",
                 "Aufgabe, Status, Frist"],
        "city": ["Heilbronn", "Stuttgart", "München", "Berlin", "Hamburg", "Köln"],
    },
    "en": {
        "doc": ["this text", "the meeting notes", "the report", "this email", "the project plan", "this paragraph"],
        "group": ["the sales team", "all employees", "the management team", "the IT department", "the project team"],
        "topic": ["the annual inventory", "the summer party", "the new travel policy", "the trade fair",
                  "the onboarding process", "the customer survey", "the office move", "the budget review"],
        "concept": ["cash flow", "depreciation", "agile project management", "machine learning",
                    "GDPR", "two-factor authentication"],
        "kpi": ["the contribution margin", "the return on investment", "the inventory turnover"],
        "dept": ["sales", "the warehouse", "marketing", "production"],
        "cols": ["month, revenue, costs", "department, budget, actual", "task, status, deadline"],
        "city": ["Munich", "Berlin", "London", "Vienna"],
    },
}


def _make_template_prompts(n: int, seed: int = SEED) -> pd.DataFrame:
    """Fill the templates with random harmless details (deterministic with the seed)."""
    rng = random.Random(seed)
    rows = []
    templates = [("de", t) for t in TEMPLATES_DE] + [("en", t) for t in TEMPLATES_EN]
    # itertools.cycle goes through the templates again and again, so all templates are used equally
    for i, (lang, template) in zip(range(n * 3), itertools.cycle(templates)):
        f = FILLERS[lang]
        text = template.format(
            doc=rng.choice(f["doc"]), group=rng.choice(f["group"]), topic=rng.choice(f["topic"]),
            concept=rng.choice(f["concept"]), kpi=rng.choice(f["kpi"]), dept=rng.choice(f["dept"]),
            cols=rng.choice(f["cols"]), city=rng.choice(f["city"]), n=rng.randint(3, 10),
            date=f"{rng.randint(1, 28)}.{rng.randint(1, 12)}." if lang == "de" else rng.choice(
                ["March 3", "May 12", "June 30", "October 1", "December 9"]),
            amount=f"{rng.randint(1, 50) * 1000:,} EUR".replace(",", ".") if lang == "de"
            else f"EUR {rng.randint(1, 50) * 1000:,}",
        )
        rows.append({"source_text": text, "language": lang, "group": f"template_{templates.index((lang, template))}",
                     "source": "templates"})
    df = pd.DataFrame(rows).drop_duplicates("source_text")
    return df.head(n)


# =========================================================================== build
def build() -> pd.DataFrame:
    oasst = _load_oasst()
    n_raw = len(oasst)

    # 1) length filter, 2) PII filter, 3) remove duplicates
    oasst = oasst[oasst["source_text"].str.len().between(MIN_CHARS, MAX_CHARS)]
    n_len = len(oasst)
    oasst = oasst[~oasst["source_text"].apply(_looks_like_pii)]
    n_pii = len(oasst)
    oasst = oasst.drop_duplicates("source_text")

    # Cap per language: shuffle once (deterministic with the seed), then keep the first N per language
    oasst = oasst.sample(frac=1, random_state=SEED).groupby("language").head(MAX_PER_LANGUAGE)

    # Split by conversation id -> deterministic and the same for everybody
    oasst["split"] = oasst["group"].map(_bucket)

    templates = _make_template_prompts(N_TEMPLATE_PROMPTS)
    templates["split"] = "train"          # generated prompts are ONLY for training

    df = pd.concat([oasst, templates], ignore_index=True)

    # Never use a prompt that is also in our own test prompts
    if OWN_PROMPTS_PATH.exists():
        own = set(pd.read_csv(OWN_PROMPTS_PATH)["prompt"].map(_normalise))
        overlap = df["source_text"].map(_normalise).isin(own)
        if overlap.any():
            print(f"[info] removed {overlap.sum()} prompts that are also in own_prompts.csv")
        df = df[~overlap]

    df = df.sort_values(["source", "language", "source_text"]).reset_index(drop=True)
    df.insert(0, "id", [f"harmless_{i:06d}" for i in range(len(df))])
    df["decision"] = "allow"
    df = df[["id", "source_text", "language", "decision", "source", "split"]]

    print(f"oasst1 first user messages ({'/'.join(LANGUAGES)}): {n_raw:,}")
    print(f"  after length filter ({MIN_CHARS}-{MAX_CHARS} chars): {n_len:,}")
    print(f"  after PII filter:                     {n_pii:,}")
    print(f"template prompts (train only):          {len(templates):,}")
    print(pd.crosstab([df["source"], df["language"]], df["split"], margins=True))
    return df


def harmless_fingerprint(path: Path = OUTPUT_PATH) -> str:
    """Short hash of harmless_prompts.csv (log it in MLflow, like split_fingerprint)."""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()[:12]


def load_harmless(split: str | None = None) -> pd.DataFrame:
    """Load the harmless prompts, optionally only one split ("train", "validation", "test")."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"{OUTPUT_PATH} is missing. Run `python src/harmless_data.py` first.")
    df = pd.read_csv(OUTPUT_PATH)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


if __name__ == "__main__":
    result = build()
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result):,} prompts to {OUTPUT_PATH}")
    print("Fingerprint:", harmless_fingerprint())
