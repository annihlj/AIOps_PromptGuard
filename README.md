# AIOps_PromptGuard
PromptGuard is a machine learning project that checks employee prompts for sensitive information before they are sent to an external AI service.

## Goal
Help IT security and compliance teams reduce accidental disclosure of personal data and credentials.

## How it works
The system detects sensitive text and applies a defined policy to return:

- **Allow:** No relevant sensitive information detected.
- **Warn:** Review or edit the prompt before sending.
- **Block:** Content violates the defined policy.

## Project approach
1. Prepare selected categories from datasets. (**ai4privacy/pii-masking-200k** ...)
2. Build a regex baseline for emails, phone numbers and bank details.
3. Train and evaluate a small Named Entity Recognition (NER) model.
4. Track experiments with **MLflow**.
5. Expose the selected model through **FastAPI**.

## Evaluation
Compare detection quality, missed sensitive content, false alarms and response times.

## Tech stack
Python; Jupyter; Hugging Face; MLflow; FastAPI; GitHub
