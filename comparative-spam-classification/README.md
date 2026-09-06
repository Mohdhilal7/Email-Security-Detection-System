# Comparative Spam Classification: TF-IDF vs DistilBERT

A complete implementation of the assignment **“Comparative Study of Traditional Machine Learning and Transformer Models for Spam Classification.”**

The project compares:
- TF-IDF + Logistic Regression
- TF-IDF + Multinomial Naive Bayes
- DistilBERT fine-tuned with a **custom PyTorch training loop** (no Hugging Face Trainer)

## Assignment coverage

| Requirement | Implementation |
|---|---|
| Required Kaggle dataset | `ashfakyeafi/spam-email-classification` |
| Dataset exploration | `src/data_utils.py` |
| Preprocessing + justification | `src/data_utils.py` + report |
| Train/validation/test split | Stratified 80/10/10 |
| TF-IDF | `src/traditional_ml.py` |
| Logistic Regression | `src/traditional_ml.py` |
| Naive Bayes | `src/traditional_ml.py` |
| Accuracy / Precision / Recall / F1 | `src/evaluation.py` |
| Confusion matrices | `src/visualize.py` |
| DistilBERT | `src/transformer_model.py` |
| PyTorch-only custom training loop | `src/transformer_model.py` |
| Training behaviour | loss/validation curves |
| Computational cost | training-time comparison |
| Matplotlib + Seaborn | `src/visualize.py` |
| Technical report | `report/technical_report.md` |
| Limitations | report section |

## Project structure

```text
comparative-spam-classification/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── README.md
├── models/
├── results/
├── report/
├── scripts/
├── src/
├── tests/
├── .github/workflows/ci.yml
├── .gitignore
└── requirements.txt
```

## 1. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## 2. Get the required dataset

The assignment specifies:

https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

You can first try the automated downloader:

```bash
python scripts/download_dataset.py
```

If that fails, download the CSV from Kaggle manually and place it under:

```text
data/raw/
```

The loader automatically searches nested CSV files and detects common text/label column names. It also supports string labels such as `spam`/`ham` and binary labels.

You can inspect the dataset before training:

```bash
python -m src.data_utils
```

## 3. Run the complete experiment

```bash
python -m src.run_all
```

This runs dataset audit, stratified splitting, Logistic Regression, Multinomial Naive Bayes, DistilBERT fine-tuning, evaluation, all required plots, and computational-cost comparison.

For a fast development smoke test:

```bash
python -m src.run_all --epochs 1 --max_train_samples 2000 --max_eval_samples 500
```

Use the full dataset for the final submission.

## 4. Individual commands

```bash
python -m src.traditional_ml
python -m src.transformer_model
python -m src.visualize
pytest -q
```

## 5. Generated output

```text
results/
├── dataset_audit.json
├── metrics.csv
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_naive_bayes.png
├── confusion_matrix_distilbert.png
├── metric_comparison.png
├── transformer_training_curve.png
└── training_time_comparison.png
```

Models are saved under `models/`.

## Methodology

### Traditional ML
A light normalization pipeline removes HTML, normalizes URLs/emails, lowercases text, and collapses whitespace. TF-IDF uses unigrams and bigrams with a bounded vocabulary. Stop-word removal is intentionally not mandatory because common words can still carry useful spam/ham context.

### Transformer
DistilBERT receives minimally normalized text so contextual information is preserved. The classifier is fine-tuned with PyTorch using AdamW, a linear warmup/decay scheduler, gradient clipping, dropout, validation monitoring, and early stopping. No Hugging Face `Trainer` API is used.

## Report

Open `report/technical_report.md` after running the experiment. Replace the result placeholders with the generated values. The report covers methodology, design decisions, experimental setup, results, comparative discussion, computational cost, and limitations.

## Important

Final metric values depend on actually running the experiment on the required Kaggle dataset and available hardware. Do not invent benchmark values in the report.
