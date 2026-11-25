# Logistic Regression – a classifier that gives probabilities

## The problem

After [TF-IDF](01_tfidf.md), every prompt is a long list of numbers. Now we need a model that looks
at these numbers and decides: **allow**, **warn** or **block**.

## The idea in one sentence

> Logistic Regression gives every text piece a **weight** (how much it points to a class), adds up
> the weights of a prompt and turns the sum into a **probability** for each class.

## Step by step

1. **Learn weights.** During training the model sees many prompts with the correct answer. It learns,
   for example:
   - pieces like `IBAN`, `Passw`, `PIN` → strong weight towards **block**
   - pieces like `@`, `.de`, `Str.` → weight towards **warn**
   - pieces like `fasse`, `summar` → weight towards **allow**
2. **Add up.** For a new prompt, the model multiplies each TF-IDF number with its weight and adds
   everything up. This gives one score per class.
3. **Turn into probabilities.** A mathematical function (*softmax*) turns the scores into percentages
   that add up to 100 %.

Example result for "Meine IBAN ist DE89 3704 0044 0532 0130 00":

| Class | Probability |
|---|---|
| allow | 2 % |
| warn | 8 % |
| block | **90 %** |

Despite the name "regression", it is a **classification** model.

## The two levers we tuned

**1. Regularisation `C`** – *How much may the model trust the training data?*
- small `C` → simple model, careful, may miss details
- large `C` → detailed model, may learn the training data by heart (*overfitting*)
- Our best value: **C = 30**

**2. Block threshold** – *From which probability do we block?*
Normally the class with the highest probability wins. For safety we block **already at 20 %** block
probability (threshold 0.2). This way we miss fewer dangerous prompts, but we accept a few more
false alarms.

We also use `class_weight="balanced"`: rare classes count more during training, so the model does
not ignore them.

## Our results (validation set)

| Version | Macro F1 | Recall block |
|---|---|---|
| Baseline (C = 1, no threshold) | 0.870 | 0.918 |
| Tuned (C = 30, threshold 0.2) | 0.864 | **0.955** |

With tuning, the model reaches our target of **at least 95 %** of block prompts blocked, in about
3 ms per prompt.

## Strengths and limits

| ✅ Strengths | ⚠️ Limits |
|---|---|
| Fast to train and to use | Only as good as the TF-IDF numbers (no context) |
| Gives probabilities → threshold is easy to set | Only "linear": weights are just added up |
| Easy to explain: we can show the top pieces per class | Combinations of pieces are not understood |
