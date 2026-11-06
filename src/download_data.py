"""Download the ai4privacy/pii-masking-200k dataset files into data/.

Usage (from the project folder, with the venv activated):
    python src/download_data.py
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ai4privacy/pii-masking-200k"
FILES = [
    "english_pii_43k.jsonl",
    "french_pii_62k.jsonl",
    "german_pii_52k.jsonl",
    "italian_pii_50k.jsonl",
]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for filename in FILES:
        target = DATA_DIR / filename
        if target.exists():
            print(f"[skip] {filename} already exists")
            continue
        print(f"[download] {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=DATA_DIR,
        )
    print(f"Done. Files are in {DATA_DIR}")


if __name__ == "__main__":
    main()
