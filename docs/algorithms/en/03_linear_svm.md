# Linear SVM – a classifier that draws the widest possible border

## The problem

Same as for [Logistic Regression](02_logistic_regression.md): every prompt is a list of numbers
from [TF-IDF](01_tfidf.md), and we want to decide **allow**, **warn** or **block**.

## The idea in one sentence

> A Linear Support Vector Machine (SVM) draws a **border** between the classes and places it so that
> the **gap to the nearest examples on both sides is as wide as possible**.

## A picture in your head

Imagine red dots (*block*) and green dots (*allow*) on a sheet of paper.
There are many straight lines that separate them. The SVM picks the line with the **widest empty
street** between the two groups.

```
   green ●  ●                 
      ●   ●     ┆           ┆         
     ●  ●       ┆  border   ┆   ▲ ▲      
                ┆     │     ┆  ▲  ▲ red
                ┆     │     ┆ ▲   ▲      
                ← wide gap →
```

Why? A wide gap means the model is **robust**: a new prompt that looks a bit different still lands
on the correct side.

The examples closest to the border are called **support vectors**. Only they decide where the border
is – hence the name.

With three classes, scikit-learn trains **one border per class** ("this class vs. all others") and
takes the class with the highest score.

## Score instead of probability

Logistic Regression says "90 % block". The SVM gives a **score** instead:
- score > 0 → on the block side of the border
- score < 0 → on the other side
- the further away from 0, the more certain

## The two levers we tuned

**1. Regularisation `C`** – *How strict is the border?*
- small `C` → wide gap, allows some mistakes in training (more robust)
- large `C` → tries to classify every training example correctly (risk of overfitting)
- Our best value: **C = 1.0**

**2. Block threshold** – *From which score do we block?*
We block already at a score of **−0.4**, i.e. slightly before the border. Like the 20 % threshold
for Logistic Regression, this catches more dangerous prompts and accepts a few more false alarms.

## Our results (validation set)

| Version | Macro F1 | Recall block | Latency p95 |
|---|---|---|---|
| Linear SVM tuned | 0.865 | **0.956** | 3.4 ms |
| Linear SVM + harmless prompts | 0.862 | **0.957** | 3.5 ms |

The SVM is slightly better than Logistic Regression at catching block prompts, but the difference is
small. Both use the same TF-IDF numbers, so they have the same limits.

## Logistic Regression vs. Linear SVM

| | Logistic Regression | Linear SVM |
|---|---|---|
| Output | probability (0–100 %) | score (distance to border) |
| Goal during training | probabilities as correct as possible | border with widest gap |
| Threshold | easy to explain ("block from 20 %") | less intuitive ("block from −0.4") |
| Speed | very fast | very fast |
| Our recall block | 0.955 | 0.956 |

## Strengths and limits

| ✅ Strengths | ⚠️ Limits |
|---|---|
| Works very well with many features (300,000 pieces) | No real probabilities |
| Robust thanks to the wide gap | Only straight borders (linear) |
| Fast | Same as TF-IDF: no understanding of context |
