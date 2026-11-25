"""Short prompts for all three decisions (allow, warn and block).

WHY THIS FILE EXISTS
--------------------
After adding the harmless prompts (src/harmless_data.py), our models let 15 to
20 of 60 warn prompts through (names and e-mail addresses in short requests).
The reason is the writing style of our training data:

    ai4privacy texts (all warn / block)   ~190 characters, formal documents
    harmless prompts (all allow)          ~70 characters, short requests
    our own test prompts (all classes)    ~75 characters, short requests

So the model learned a shortcut: "short request = allow". It never saw a short
request that contains a name, an e-mail address or an IBAN.

This file builds exactly those missing examples: short requests in German and
English, in ALL THREE classes, so that the style no longer tells the decision
and the model has to look at the content ("is there a name? an IBAN?").

HOW THE PROMPTS ARE BUILT
-------------------------
1. Sentence frames (FRAMES below), e.g.
       "Schreib eine Mail an {recipient} wegen {topic}."
   Every frame has slots. Each slot KIND can be filled with an allow, a warn or
   a block value, for example the kind "recipient":
       allow -> "das Vertriebsteam"          (no personal data)
       warn  -> "anna.berg@example.com"      (EMAIL)
   The SAME frame is used for every class it can take, so the frame itself says
   nothing about the decision. Only the slot content does.

2. The personal data values (names, e-mails, IBANs, passwords, ...) are taken
   from the privacy_mask of the ai4privacy texts of the SAME split:
       train frames      <- values from ai4privacy train
       validation frames <- values from ai4privacy validation
       test frames       <- values from ai4privacy test
   So no value of the validation or test set is ever seen in training.

3. The decision is NOT set by hand: it is calculated with prompt_decision()
   from src/data_utils.py, i.e. with our policy (config/policy.yaml). The script
   checks that it matches the class we wanted to build.

THE SPLITS
----------
The frames are divided into three groups that do not share any frame:
    train       many frames, 2,000 prompts per class and language
    validation  other frames, 100 per class and language (200 / 200 / 200)
    test        again other frames, 100 per class and language (200 / 200 / 200)
Validation is used to choose the block threshold, so the threshold also sees
short requests with personal data. Test is only used once, in Milestone 5.
Because the validation and test frames are different from the training frames,
the result shows whether the model learned the CONTENT, not the frames.

RULES
-----
- German and English only (our product is for German companies; English is
  common in German offices too).
- The frames were written WITHOUT looking at data/own_prompts.csv, so our own
  test prompts stay an independent check. In addition the script removes any
  generated prompt that is identical to one of our own test prompts.
- About 30 % of the prompts are written casually (starting lowercase), in every
  class, so that "casual" does not become a new shortcut.
- The result is written to data/short_prompts.csv (not in Git, like all data).

USAGE
-----
    python src/short_prompts.py          # builds data/short_prompts.csv

In a notebook:
    sys.path.append("../src")
    from short_prompts import load_short_prompts, short_prompts_fingerprint
    short_train = load_short_prompts("train")
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

from data_utils import DATA_DIR, load_policy, load_split_ids, prompt_decision

OUTPUT_PATH = DATA_DIR / "short_prompts.csv"
OWN_PROMPTS_PATH = DATA_DIR / "own_prompts.csv"

# The ai4privacy file per language (only German and English are used here)
SOURCE_FILES = {"de": "german_pii_52k.jsonl", "en": "english_pii_43k.jsonl"}

# How many prompts per class and language
N_PER_CLASS = {"train": 2000, "validation": 100, "test": 100}
CLASSES = ["allow", "warn", "block"]
CASUAL_SHARE = 0.3            # share of prompts written casually (lowercase start)
MAX_VALUE_CHARS = 60          # very long values do not look like typed prompts
SEED = 42

# =========================================================================== slot kinds
# Which personal data labels can fill which slot kind, per decision.
# "generic" means: a phrase from GENERIC below, without any personal data.
# The labels are the ai4privacy labels; their decision comes from policy.yaml.
SLOT_SOURCES = {
    # (no COMPANYNAME for allow: in ai4privacy company names often look like surnames,
    #  e.g. "King - Hand", which would teach the model "surname = allow")
    "recipient": {"allow": ["generic"],
                  "warn": ["FULLNAME", "FIRSTNAME", "EMAIL", "PREFIX_LASTNAME"]},
    "contact":   {"allow": ["generic"],
                  "warn": ["PHONENUMBER", "EMAIL", "ADDRESS"]},
    "account":   {"allow": ["generic"],
                  "block": ["IBAN", "ACCOUNTNUMBER", "CREDITCARDNUMBER"]},
    "secret":    {"allow": ["generic"],
                  "block": ["PASSWORD", "PIN"]},
    "govid":     {"allow": ["generic"],
                  "block": ["SSN"]},
    "device":    {"allow": ["generic"],
                  "warn": ["IPV4", "USERNAME"]},
    "topic":     {"allow": ["generic"]},
}

# Generic phrases (no personal data) per slot kind and language
GENERIC = {
    "de": {
        "recipient": ["das Vertriebsteam", "die Buchhaltung", "alle Abteilungsleiter", "das Projektteam",
                      "den Einkauf", "die IT-Abteilung", "das Lager", "die Geschäftsführung", "das Marketing",
                      "die Personalabteilung", "das Controlling", "den Kundenservice", "die Rechtsabteilung",
                      "alle Mitarbeitenden", "das Qualitätsmanagement", "die Logistik", "den Betriebsrat",
                      "die Produktion"],
        # (only feminine / neuter phrases, so they fit every German frame unchanged)
        "contact": ["die zentrale Hotline", "das Sekretariat", "die Support-Hotline", "die Zentrale",
                    "das Service-Postfach"],
        "account": ["unser Geschäftskonto", "das hinterlegte Lieferantenkonto", "die bekannte Bankverbindung",
                    "das Firmenkonto", "die hinterlegte IBAN", "die Firmenkreditkarte"],
        "secret": ["unsere neue Passwort-Richtlinie", "die Regeln für sichere Passwörter", "ein starkes Passwort",
                   "die Vorgaben zur PIN-Länge", "die Zwei-Faktor-Anmeldung"],
        "govid": ["die Kostenstelle", "das Eintrittsdatum", "die Abteilung", "die Projektnummer",
                  "das leere Feld Sozialversicherungsnummer"],
        "device": ["dem Gäste-WLAN", "dem Druckserver", "dem Testsystem", "dem Intranet",
                   "der IP des Besprechungsraums", "dem Benutzerkonto der Praktikanten"],
        # (without article, used as "zum Thema Inventur")
        "topic": ["Inventur", "Teammeeting", "Lieferung", "offene Rechnung", "Angebot",
                  "Reisekosten", "Projektstart", "Messe", "Schulung", "Quartalsbudget",
                  "Jahresplanung", "Urlaubsplanung", "Preisanpassung", "Kundenumfrage", "Lieferantenaudit",
                  "Büroumzug", "Sommerfest", "Datenschutzschulung", "Wartungsfenster", "Vertragsverlängerung"],
    },
    "en": {
        "recipient": ["the sales team", "accounting", "all department heads", "the project team",
                      "purchasing", "the IT department", "the warehouse", "management", "marketing",
                      "HR", "controlling", "customer service", "the legal team", "all employees",
                      "quality management", "logistics", "the works council", "production"],
        "contact": ["the central hotline", "the front office", "general support", "the reception desk",
                    "the service inbox"],
        "account": ["our business account", "the stored supplier account", "the usual bank details",
                    "the company account", "the stored IBAN", "the company credit card"],
        "secret": ["our new password policy", "the rules for safe passwords", "a strong password",
                   "the PIN length rules", "two-factor login"],
        "govid": ["the cost center", "the start date", "the department", "the project number",
                  "the empty social security number field"],
        "device": ["the guest Wi-Fi", "the print server", "the test system", "the intranet",
                   "the meeting room IP", "the intern user account"],
        "topic": ["the inventory", "the team meeting", "the delivery", "the open invoice", "the offer",
                  "travel expenses", "the project start", "the trade fair", "the training", "the quarterly budget",
                  "the annual plan", "the holiday schedule", "the price change", "the customer survey",
                  "the supplier audit", "the office move", "the summer party", "the privacy training",
                  "the maintenance window", "the contract renewal"],
    },
}

# =========================================================================== frames
# Each frame is a sentence with {slots}; the slot name is its kind, e.g. {recipient}
# or {account}. Frames are grouped by split and never shared between splits.
FRAMES = {
    "train": {
        "de": [
            "Schreib eine kurze Mail an {recipient} zum Thema {topic}.",
            "Erinnere {recipient} bitte an das Thema {topic}.",
            "Formuliere eine Absage an {recipient}.",
            "Fasse die Rückmeldung für {recipient} zum Thema {topic} zusammen.",
            "Bitte um Rückruf über {contact} zum Thema {topic}.",
            "Schreib eine Notiz: Rückfragen bitte an {contact}.",
            "Überweise den Betrag (Posten: {topic}) auf {account}.",
            "Erstelle eine Zahlungserinnerung, Zahlung bitte auf {account}.",
            "Prüf bitte, ob {secret} sicher genug ist.",
            "Schreib eine Anleitung, in der {secret} erklärt wird.",
            "Trag {govid} in das Personalformular ein.",
            "Der Zugriff von {device} funktioniert nicht, schreib ein Ticket.",
            "Schreib eine Info an {recipient}: Kontakt zum Thema {topic} ist {contact}.",
            "Bereite eine Lastschrift für {recipient} vor, Konto: {account}.",
        ],
        "en": [
            "Write a short email to {recipient} about {topic}.",
            "Please remind {recipient} about {topic}.",
            "Draft a polite rejection to {recipient}.",
            "Summarize the call with {recipient} about {topic}.",
            "Ask for a callback via {contact} about {topic}.",
            "Write a note: questions go to {contact}.",
            "Transfer the amount for {topic} to {account}.",
            "Create a payment reminder, payment to {account}.",
            "Check if {secret} is strong enough.",
            "Write instructions that explain {secret}.",
            "Enter {govid} in the HR form.",
            "Access from {device} is not working, write a ticket.",
            "Send an update to {recipient}: contact for {topic} is {contact}.",
            "Prepare a direct debit for {recipient}, account: {account}.",
        ],
    },
    "validation": {
        "de": [
            "Kannst du {recipient} kurz informieren, Thema {topic}?",
            "Notiere als Ansprechpartner {contact}.",
            "Buch die Gebühr über {account} ab.",
            "Merk dir {secret} für später.",
            "Leg {govid} in der Akte ab.",
            "Prüf die Anmeldungen von {device}.",
        ],
        "en": [
            "Can you quickly let {recipient} know about {topic}?",
            "Note {contact} as the contact person.",
            "Charge the fee to {account}.",
            "Keep {secret} in mind for later.",
            "File {govid} in the record.",
            "Check the logins from {device}.",
        ],
    },
    "test": {
        "de": [
            "Leite die Unterlagen zum Thema {topic} an {recipient} weiter.",
            "Die Kontaktdaten sind {contact}, formuliere eine Visitenkarte.",
            "Begleich die Rechnung über {account}.",
            "Speichere {secret} in der Dokumentation.",
            "Ergänze {govid} im Antrag.",
            "Sperr den Zugang von {device}.",
        ],
        "en": [
            "Forward the documents about {topic} to {recipient}.",
            "The contact details are {contact}, draft a business card.",
            "Settle the invoice via {account}.",
            "Save {secret} in the documentation.",
            "Add {govid} to the application.",
            "Block access from {device}.",
        ],
    },
}

# How a value is written into a prompt. Real prompts usually name what it is
# ("das Passwort ...", "Konto ..."). The same words also appear in the generic
# allow phrases above ("die Passwort-Richtlinie", "das Firmenkonto"), so the word
# alone does not give away the decision: only the value does.
WRAP = {
    "de": {"PASSWORD": ["das Passwort {v}", "das Kennwort {v}", "mein Passwort {v}"],
           "PIN": ["die PIN {v}", "PIN {v}"],
           "IBAN": ["IBAN {v}", "{v}"],
           "ACCOUNTNUMBER": ["Konto {v}", "Kontonummer {v}"],
           "CREDITCARDNUMBER": ["Karte {v}", "Kreditkarte {v}"],
           "SSN": ["die Sozialversicherungsnummer {v}", "SV-Nummer {v}"],
           "IPV4": ["der IP {v}", "IP {v}"],
           "USERNAME": ["Benutzer {v}", "dem Account {v}"],
           "PHONENUMBER": ["Tel. {v}", "{v}"]},
    "en": {"PASSWORD": ["the password {v}", "my password {v}", "password {v}"],
           "PIN": ["the PIN {v}", "PIN {v}"],
           "IBAN": ["IBAN {v}", "{v}"],
           "ACCOUNTNUMBER": ["account {v}", "account number {v}"],
           "CREDITCARDNUMBER": ["card {v}", "credit card {v}"],
           "SSN": ["SSN {v}", "social security number {v}"],
           "IPV4": ["IP {v}", "the IP {v}"],
           "USERNAME": ["user {v}", "the account {v}"],
           "PHONENUMBER": ["phone {v}", "{v}"]},
}

_SLOT = re.compile(r"\{(\w+)\}")


# =========================================================================== values
def load_values(split: str) -> dict[str, dict[str, list[str]]]:
    """All personal data values of ONE ai4privacy split, per language and label.

    Reads the jsonl files line by line (little memory) and keeps only texts whose
    id belongs to the wanted split in data/splits.csv.
    """
    ids = load_split_ids()
    wanted = set(ids.loc[ids["split"] == split, "id"])
    values: dict[str, dict[str, set[str]]] = {}
    for language, filename in SOURCE_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run `python src/download_data.py` first.")
        per_label: dict[str, set[str]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("id") not in wanted:
                    continue
                for entity in row.get("privacy_mask", []):
                    value = str(entity.get("value", "")).strip()
                    # at least 2 characters (drops broken values like "x"), no line breaks
                    if 2 <= len(value) <= MAX_VALUE_CHARS and "\n" not in value:
                        per_label.setdefault(entity["label"], set()).add(value)
        values[language] = per_label
    # sorted lists, so that random choices are the same on every computer
    return {lang: {label: sorted(v) for label, v in labels.items()} for lang, labels in values.items()}


def fill_slot(kind: str, wanted: str, language: str, values: dict, rng: random.Random):
    """Returns (text, entities) for one slot of the given kind and decision.

    entities is the list of personal data in the text, in privacy_mask format,
    so that prompt_decision() can calculate the decision with our policy.
    """
    source = rng.choice(SLOT_SOURCES[kind][wanted])
    v = values[language]
    if source == "generic":
        return rng.choice(GENERIC[language][kind]), []
    if source == "FULLNAME":
        first, last = rng.choice(v["FIRSTNAME"]), rng.choice(v["LASTNAME"])
        return f"{first} {last}", [{"label": "FIRSTNAME", "value": first}, {"label": "LASTNAME", "value": last}]
    if source == "PREFIX_LASTNAME":
        prefix = rng.choice(["Frau", "Herrn"] if language == "de" else ["Ms.", "Mr."])
        last = rng.choice(v["LASTNAME"])
        return f"{prefix} {last}", [{"label": "PREFIX", "value": prefix}, {"label": "LASTNAME", "value": last}]
    if source == "ADDRESS":
        street, number = rng.choice(v["STREET"]), rng.choice(v["BUILDINGNUMBER"])
        zipcode, city = rng.choice(v["ZIPCODE"]), rng.choice(v["CITY"])
        text = f"{street} {number}, {zipcode} {city}"
        return text, [{"label": "STREET", "value": street}, {"label": "BUILDINGNUMBER", "value": number},
                      {"label": "ZIPCODE", "value": zipcode}, {"label": "CITY", "value": city}]
    value = rng.choice(v[source])
    text = rng.choice(WRAP[language].get(source, ["{v}"])).format(v=value)
    return text, [{"label": source, "value": value}]


def slot_plan(kinds: list[str], wanted: str, rng: random.Random) -> list[str] | None:
    """Which decision each slot gets, so that the whole prompt has the wanted decision.

    allow: every slot gets an allow value.
    warn / block: one slot that can take that decision gets it; the other slots get
    allow (or, for block, sometimes warn: e.g. a name plus an IBAN = block).
    Returns None if the frame cannot reach the wanted decision.
    """
    if wanted == "allow":
        return ["allow"] * len(kinds)
    able = [i for i, k in enumerate(kinds) if wanted in SLOT_SOURCES[k]]
    if not able:
        return None
    main = rng.choice(able)
    plan = []
    for i, k in enumerate(kinds):
        if i == main:
            plan.append(wanted)
        elif wanted == "block" and "warn" in SLOT_SOURCES[k] and rng.random() < 0.3:
            plan.append("warn")
        else:
            plan.append("allow")
    return plan


def casual(text: str) -> str:
    """Casual style: first letter lowercase and no final full stop."""
    text = text[0].lower() + text[1:]
    return text[:-1] if text.endswith(".") else text


def build_split(split: str, policy: dict, rng: random.Random) -> pd.DataFrame:
    values = load_values(split)
    rows = []
    for language in SOURCE_FILES:
        frames = FRAMES[split][language]
        for wanted in CLASSES:
            # frames that can produce this decision
            usable = [(i, f) for i, f in enumerate(frames)
                      if slot_plan(_SLOT.findall(f), wanted, random.Random(0)) is not None]
            seen, attempts = set(), 0
            while len(seen) < N_PER_CLASS[split] and attempts < N_PER_CLASS[split] * 50:
                attempts += 1
                frame_index, frame = rng.choice(usable)
                kinds = _SLOT.findall(frame)
                plan = slot_plan(kinds, wanted, rng)
                parts, entities = {}, []
                for kind, decision in zip(kinds, plan):
                    text, ents = fill_slot(kind, decision, language, values, rng)
                    parts[kind] = text
                    entities += ents
                prompt = frame.format(**parts)
                if rng.random() < CASUAL_SHARE:
                    prompt = casual(prompt)
                decision = prompt_decision(entities, policy)     # our policy decides, not we
                if decision != wanted or prompt in seen:
                    continue
                seen.add(prompt)
                rows.append({"source_text": prompt, "language": language, "decision": decision,
                             "split": split, "frame_id": f"{split}_{language}_{frame_index:02d}",
                             "labels": ",".join(sorted({e["label"] for e in entities}))})
            if len(seen) < N_PER_CLASS[split]:
                print(f"[warning] only {len(seen)} unique {wanted} prompts for {split}/{language}")
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    policy = load_policy()
    df = pd.concat([build_split(s, policy, rng) for s in N_PER_CLASS], ignore_index=True)

    # Never use a prompt that is also one of our own test prompts
    if OWN_PROMPTS_PATH.exists():
        own = set(pd.read_csv(OWN_PROMPTS_PATH)["prompt"].str.lower().str.strip())
        overlap = df["source_text"].str.lower().str.strip().isin(own)
        if overlap.any():
            print(f"[info] removed {overlap.sum()} prompts that are also in own_prompts.csv")
        df = df[~overlap]

    df = df.sort_values(["split", "language", "decision", "source_text"]).reset_index(drop=True)
    df.insert(0, "id", [f"short_{i:06d}" for i in range(len(df))])

    print(pd.crosstab([df["split"], df["language"]], df["decision"], margins=True))
    print("\nMedian length in characters:")
    print(df.groupby("decision")["source_text"].apply(lambda s: s.str.len().median()))
    print("\nSPOT CHECK: 5 random prompts per class (train)")
    for decision in CLASSES:
        sample = df[(df["split"] == "train") & (df["decision"] == decision)].sample(5, random_state=SEED)
        for _, row in sample.iterrows():
            print(f"  [{decision}] {row['source_text']}   ({row['labels'] or 'no personal data'})")
    return df


def short_prompts_fingerprint(path: Path = OUTPUT_PATH) -> str:
    """Short hash of short_prompts.csv (log it in MLflow, like split_fingerprint)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def load_short_prompts(split: str | None = None) -> pd.DataFrame:
    """Load the short prompts, optionally only one split ("train", "validation", "test")."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"{OUTPUT_PATH} is missing. Run `python src/short_prompts.py` first.")
    df = pd.read_csv(OUTPUT_PATH)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


if __name__ == "__main__":
    result = build()
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} prompts to {OUTPUT_PATH}")
    print("Fingerprint:", short_prompts_fingerprint())
