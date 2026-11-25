# TF-IDF – turning text into numbers

## The problem

A computer cannot read text. A machine learning model only works with **numbers**.
So before we can classify a prompt as *allow*, *warn* or *block*, we have to turn every prompt into a
list of numbers. TF-IDF is the method we use for this.

## The idea in one sentence

> TF-IDF gives a high number to text pieces that appear **often in this prompt** but **rarely in all
> other prompts**, because these pieces say the most about the prompt.

## The two parts

**TF = Term Frequency** – *How often does a piece appear in this prompt?*
If "IBAN" appears in a prompt, that is a hint. If it appears three times, the hint is a bit stronger.

**IDF = Inverse Document Frequency** – *How rare is the piece across all prompts?*
Pieces like "the" or "and" appear in almost every prompt. They tell us nothing, so they get a **low**
weight. A piece like "IBAN" or "Passwort" is rare, so it gets a **high** weight.

**TF × IDF** = the final number for each piece in each prompt.

## Small example

| Piece | Appears in this prompt? | Appears in many prompts? | TF-IDF |
|---|---|---|---|
| "please" | yes | yes, very often | low |
| "summarise" | yes | quite often | medium |
| "IBAN" | yes | rarely | **high** |

Every prompt becomes a long list (a *vector*) with one number per piece. Most numbers are 0, because
a single prompt only contains a few of all possible pieces.

## What we do in PromptGuard

We do not cut the text into whole words. We cut it into **short letter sequences of 2 to 5
characters** (character n-grams), for example `"IBAN"` → `IB`, `BA`, `AN`, `IBA`, `BAN`, `IBAN`.

Why letters instead of words?

- **Spelling and typos:** "Passwort", "passwort" and "Passwrt" still share many pieces.
- **Several languages:** works the same for German, English, French and Italian.
- **Patterns in numbers:** an IBAN like `DE89 3704…` or an e-mail with `@` and `.de` produces typical
  pieces, even if the model has never seen this exact number.

Our settings (scikit-learn `TfidfVectorizer`):

| Setting | Value | Meaning in simple words |
|---|---|---|
| `analyzer` | `char_wb` | letter pieces, only inside word boundaries |
| `ngram_range` | `(2, 5)` | pieces with 2 to 5 characters |
| `min_df` | `5` | ignore pieces that appear in fewer than 5 prompts (noise) |
| `max_features` | `300,000` | keep at most the 300,000 most frequent pieces |
| `sublinear_tf` | `True` | 10 repetitions count more than 1, but not 10 times as much |

## Strengths and limits

| ✅ Strengths | ⚠️ Limits |
|---|---|
| Very fast (a few milliseconds) | Does not understand meaning or context |
| Easy to explain | Word order is mostly lost ("bag of pieces") |
| Works well with typos and many languages | A name like "Anna" looks like any other word |

The last point explains one of our results: TF-IDF cannot tell whether "Anna" is a person in the
prompt or just a normal word. That is why some prompts with names are still let through.

**Important:** TF-IDF itself does not make a decision. It only creates the numbers. The decision is
made by the classifier afterwards: [Logistic Regression](02_logistic_regression.md) or
[Linear SVM](03_linear_svm.md).
