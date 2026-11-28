"""Helpers for the transformer models (encoder or small LLM, each + LoRA) of PromptGuard.

WHY THIS FILE EXISTS
--------------------
The transformer models are trained in Google Colab
    notebooks/train_encoder_lora_colab.ipynb   encoder (XLM-RoBERTa)
    notebooks/train_llm_lora_colab.ipynb       small LLM (Qwen3)
but their speed is measured on our laptops (src/measure_latency.py). All of them need the
same code to load the model and to turn a prompt into allow / warn / block. That code is
here, so the notebooks and the scripts cannot drift apart. Later the API (app/) can use it too.

WHAT IS SAVED AFTER TRAINING (one folder, e.g. models/xlmr_lora_benign/)
------------------------------------------------------------------------
    adapter_config.json          LoRA settings (written by peft)
    adapter_model.safetensors    the trained LoRA weights + the classification head (a few MB)
    tokenizer files              how texts are cut into tokens
    promptguard_config.json      our own info: base model + revision, classes, max_length,
                                 block threshold, fingerprints of the data
                                 (LLM runs also: model_family = "llm", pad_token_id, quantization)
The big base model (xlm-roberta-base about 1 GB, Qwen3-0.6B about 1.2 GB) is NOT saved. It is
downloaded from Hugging Face at exactly the revision (commit hash) in promptguard_config.json.

ENCODER AND LLM: WHAT IS DIFFERENT WHEN LOADING
-----------------------------------------------
Both are loaded with AutoModelForSequenceClassification (one label per text). Two differences:
1. An LLM reads from left to right, so it classifies with the LAST real token of the text.
   To find it in a padded batch, the model must know the padding token (pad_token_id).
2. We load the LLM in float32 (full precision). The encoder is loaded exactly as before,
   so the encoder runs stay reproducible. Older config files have no "model_family":
   they are encoders.

TYPICAL USAGE
-------------
    from encoder_utils import load_model, predict_proba, predict_with_threshold
    model, tokenizer, cfg = load_model("models/xlmr_lora_benign")
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
#   target_modules:  the attention layers that get LoRA matrices
#   modules_to_save: layers that are trained COMPLETELY (not with LoRA): the new
#                    classification head, which does not exist in the pre-trained model
#                    (called "classifier" in XLM-R, "score" in Qwen3)
_QWEN3_LAYERS = {"target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],   # all 4 attention projections
                 "modules_to_save": ["score"]}
LORA_LAYERS = {
    # "query" and "value" is the setting of the original LoRA paper
    "FacebookAI/xlm-roberta-base": {"target_modules": ["query", "value"],
                                    "modules_to_save": ["classifier"]},
    # for a later comparison run:
    "microsoft/mdeberta-v3-base": {"target_modules": ["query_proj", "value_proj"],
                                   "modules_to_save": ["classifier", "pooler"]},
    # small LLMs (notebooks/train_llm_lora_colab.ipynb)
    "Qwen/Qwen3-0.6B": _QWEN3_LAYERS,
    "Qwen/Qwen3-1.7B": _QWEN3_LAYERS,
}


def load_tokenizer(name_or_dir, revision: str | None = None, model_family: str = "encoder"):
    """The tokenizer of the base model (training) or of a saved model folder (later use).

    An LLM tokenizer sometimes has no padding token, because LLMs normally generate one
    text at a time. For a classifier we put several texts of different length into one batch,
    so we need one: then we use the end-of-text token.
    """
    tokenizer = AutoTokenizer.from_pretrained(name_or_dir, revision=revision)
    if model_family == "llm" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_model: str, revision: str, classes, model_family: str = "encoder",
                    pad_token_id: int | None = None, quantization: str | None = None):
    """The pre-trained model with a NEW classification head (random weights, len(classes) outputs).

    Used for training (notebooks) and for loading a trained model (load_model below).
    quantization="4bit" (QLoRA, only for training on a GPU): the frozen base weights are stored
    with 4 bits instead of 32, so a bigger model fits on the Colab GPU. Needs bitsandbytes
    (requirements-train.txt).
    """
    kwargs = {
        "revision": revision,
        "num_labels": len(classes),
        "id2label": dict(enumerate(classes)),
        "label2id": {c: i for i, c in enumerate(classes)},
    }
    if model_family == "llm":
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",              # 4-bit format made for neural network weights
                bnb_4bit_use_double_quant=True,         # saves a bit more memory
                bnb_4bit_compute_dtype=torch.float16,   # calculations in 16 bit (the T4 has no bfloat16)
            )
            kwargs["device_map"] = {"": 0}              # 4-bit weights only work on the GPU
        else:
            # Qwen3 is stored in bfloat16; we calculate in float32 (on the GPU the Trainer
            # adds fp16 itself, on the CPU float32 is the fast and exact choice)
            kwargs["dtype"] = torch.float32
    # The encoder gets no extra arguments: exactly the same call as in runs 1 and 2.
    model = AutoModelForSequenceClassification.from_pretrained(base_model, **kwargs)
    if model_family == "llm":
        # the LLM classifies with the last token that is not padding, so it must know the padding token
        model.config.pad_token_id = pad_token_id
        model.config.use_cache = False                  # cache is only useful for generating text
    return model


def load_model(model_dir: str | Path, device: str = "cpu", merge: bool = True):
    """Load a saved PromptGuard model (base model + LoRA adapter), encoder or LLM.

    merge=True adds the LoRA weights into the base model once ("merge_and_unload").
    The predictions stay the same, but the model gets a bit faster, because the
    LoRA matrices no longer have to be calculated separately for every prompt.

    A model trained with 4-bit (QLoRA) is loaded here WITHOUT 4-bit: 4-bit only works on a
    GPU, and we use the model on the CPU. The decisions can differ for a few prompts.

    Returns (model, tokenizer, cfg), cfg = content of promptguard_config.json.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    family = cfg.get("model_family", "encoder")     # older config files: always an encoder

    tokenizer = load_tokenizer(model_dir, model_family=family)
    # 1) the pre-trained base model, at the same revision as in training.
    #    transformers prints a warning that the classifier is "newly initialized" / MISSING:
    #    that is expected, the trained classifier is loaded in step 2.
    base = load_base_model(cfg["base_model"], cfg["base_model_revision"], cfg["classes"],
                           model_family=family, pad_token_id=cfg.get("pad_token_id", tokenizer.pad_token_id))
    # 2) put our trained LoRA weights and classification head on top
    model = PeftModel.from_pretrained(base, model_dir)
    if merge:
        model = model.merge_and_unload()
    model.to(device).eval()      # eval() switches off dropout (we only predict)
    return model, tokenizer, cfg


# The old name, used by train_encoder_lora_colab.ipynb and the scripts
load_encoder = load_model


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
