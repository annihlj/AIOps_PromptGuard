"""Helpers for the encoder model (transformer + LoRA) of PromptGuard.

WHY THIS FILE EXISTS
--------------------
The encoder model is trained in Google Colab (notebooks/train_encoder_lora_colab.ipynb),
but its speed is measured on our laptops (src/measure_latency.py). Both need the same
code to load the model and to turn a prompt into allow / warn / block. That code is here,
so the notebook and the script cannot drift apart. Later the API (app/) can use it too.

WHAT IS SAVED AFTER TRAINING (one folder, e.g. models/xlmr_lora_benign/)
------------------------------------------------------------------------
    adapter_config.json          LoRA settings (written by peft)
    adapter_model.safetensors    the trained LoRA weights + the classification head (a few MB)
    tokenizer files              how texts are cut into tokens
    promptguard_config.json      our own info: base model + revision, classes, max_length,
                                 block threshold, fingerprints of the data
The big base model (xlm-roberta-base, about 1 GB) is NOT saved. It is downloaded from
Hugging Face at exactly the revision (commit hash) written in promptguard_config.json.

TYPICAL USAGE
-------------
    from encoder_utils import load_encoder, predict_proba, predict_with_threshold
    model, tokenizer, cfg = load_encoder("models/xlmr_lora_benign")
    proba = predict_proba(model, tokenizer, ["Fasse den Text zusammen"], cfg["max_length"])
    decision = predict_with_threshold(proba, cfg["classes"], cfg["block_threshold"])
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# The order of the classes = the order of the model's outputs (output 0 = allow, ...)
CLASSES = ["allow", "warn", "block"]
CONFIG_FILE = "promptguard_config.json"

# LoRA adds small trainable matrices to some layers of the transformer.
# The layer names differ between model families, so we keep them here per base model.
#   target_modules:  the attention layers that get LoRA matrices ("query" and "value"
#                    is the setting of the original LoRA paper)
#   modules_to_save: layers that are trained COMPLETELY (not with LoRA): the new
#                    classification head, which does not exist in the pre-trained model
LORA_LAYERS = {
    "FacebookAI/xlm-roberta-base": {"target_modules": ["query", "value"],
                                    "modules_to_save": ["classifier"]},
    # for a later comparison run:
    "microsoft/mdeberta-v3-base": {"target_modules": ["query_proj", "value_proj"],
                                   "modules_to_save": ["classifier", "pooler"]},
}


def load_encoder(model_dir: str | Path, device: str = "cpu", merge: bool = True):
    """Load a saved PromptGuard encoder (base model + LoRA adapter).

    merge=True adds the LoRA weights into the base model once ("merge_and_unload").
    The predictions stay the same, but the model gets a bit faster, because the
    LoRA matrices no longer have to be calculated separately for every prompt.

    Returns (model, tokenizer, cfg), cfg = content of promptguard_config.json.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    classes = cfg["classes"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 1) the pre-trained base model, at the same revision as in training.
    #    transformers prints a warning that the classifier is "newly initialized":
    #    that is expected, the trained classifier is loaded in step 2.
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"],
        revision=cfg["base_model_revision"],
        num_labels=len(classes),
        id2label=dict(enumerate(classes)),
        label2id={c: i for i, c in enumerate(classes)},
    )
    # 2) put our trained LoRA weights and classification head on top
    model = PeftModel.from_pretrained(base, model_dir)
    if merge:
        model = model.merge_and_unload()
    model.to(device).eval()      # eval() switches off dropout (we only predict)
    return model, tokenizer, cfg


@torch.no_grad()                 # no gradients needed for predictions: faster, less memory
def predict_proba(model, tokenizer, texts: list[str], max_length: int = 128,
                  batch_size: int = 64) -> np.ndarray:
    """Probabilities for every text and class, shape (number of texts, 3).

    Column order = CLASSES (allow, warn, block).
    """
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        # padding=True: texts in one batch are filled up to the longest one
        # truncation: texts longer than max_length tokens are cut off
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        logits = model(**enc).logits
        # softmax turns the raw scores (logits) into probabilities that add up to 1
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def predict_with_threshold(proba: np.ndarray, classes, threshold: float | None) -> np.ndarray:
    """Same rule as in the TF-IDF notebooks.

    Block if p(block) >= threshold, otherwise the more likely of allow / warn.
    threshold=None: the class with the highest probability wins (argmax).
    """
    classes = list(classes)
    if threshold is None:
        return np.array(classes)[proba.argmax(axis=1)]
    p_block = proba[:, classes.index("block")]
    other = [c for c in classes if c != "block"]
    best_other = np.array(other)[proba[:, [classes.index(c) for c in other]].argmax(axis=1)]
    return np.where(p_block >= threshold, "block", best_other)
