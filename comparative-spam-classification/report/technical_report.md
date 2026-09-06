# Technical Report — Comparative Spam Classification

**Student:** Mohammed Hilal  
**Dataset:** Kaggle `ashfakyeafi/spam-email-classification`  
**Models:** TF-IDF + Logistic Regression, TF-IDF + Multinomial Naive Bayes, DistilBERT + PyTorch

> Run the experiment first and replace all `[GENERATED]` fields with actual results. Do not invent benchmark values.

## 1. Objective

This project compares traditional NLP with transformer-based contextual representation learning for binary Spam/Ham email classification. The traditional pipeline uses TF-IDF with Logistic Regression and Multinomial Naive Bayes. The transformer pipeline fine-tunes DistilBERT using a custom PyTorch training loop without the Hugging Face Trainer API.

## 2. Dataset Exploration and Preparation

The required Kaggle dataset is loaded from `data/raw/`. The loader reports shape, class distribution, missing/empty text, invalid labels, duplicate messages, and text-length statistics in `results/dataset_audit.json`.

The experiment uses a stratified 80/10/10 train/validation/test split. Stratification preserves the class ratio and the test set is kept untouched until final evaluation.

For TF-IDF, preprocessing removes HTML, normalizes URLs and email addresses into tokens, lowercases text, and collapses whitespace. Unigrams and bigrams are retained because short phrases can carry spam signals. For DistilBERT, preprocessing is intentionally lighter to preserve contextual information.

## 3. Traditional Machine Learning

TF-IDF is fitted only on training text and then applied to the test set. Logistic Regression uses class weighting and a high iteration limit. Multinomial Naive Bayes uses a smoothing parameter of 0.5. Both models use the same TF-IDF representation.

Evaluation includes Accuracy, Precision, Recall, F1-score and confusion matrices, with Spam treated as the positive class.

## 4. Transformer Model

DistilBERT (`distilbert-base-uncased`) is fine-tuned end-to-end with a dropout classification head. Training uses PyTorch `DataLoader`, AdamW, linear warmup/decay, gradient clipping, validation monitoring, and early stopping. No high-level `Trainer` API is used.

| Hyperparameter | Value |
|---|---:|
| Max length | 256 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 3 requested |
| Warmup ratio | 10% |
| Gradient clipping | 1.0 |
| Dropout | 0.30 |
| Early stopping patience | 2 |

## 5. Results

| Model | Accuracy | Precision | Recall | F1 | Training time (s) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] |
| Naive Bayes | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] |
| DistilBERT | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] | [GENERATED] |

Required visualizations are generated under `results/`: metric comparison, three confusion matrices, transformer training behaviour, and training-time comparison.

## 6. Comparative Discussion

The final discussion should compare the models using the measured Accuracy, Precision, Recall and F1-score rather than accuracy alone. False positives and false negatives have different operational implications in spam filtering, so the confusion matrices should be considered alongside aggregate metrics.

TF-IDF models are lightweight and fast, while DistilBERT can model contextual relationships but requires substantially more compute and memory. The best engineering choice therefore depends on the measured performance gain relative to computational cost.

**Best model:** [GENERATED]  
**Reason:** [GENERATED]

## 7. Computational Cost

**Hardware:** [GENERATED]  
**Traditional ML time:** [GENERATED]  
**DistilBERT time:** [GENERATED]

Training time is hardware-dependent and should be reported with the final submission.

## 8. Limitations

1. Results depend on the exact dataset version used.
2. DistilBERT truncates sequences beyond the configured maximum length.
3. Transformer training is more computationally expensive than TF-IDF baselines.
4. Dataset-specific language patterns may not generalize to future spam.
5. A single split introduces sampling variance.
6. The project uses email text rather than sender reputation, attachments, structured URL intelligence, or external threat intelligence.

## 9. Conclusion

The implementation provides a controlled comparison between feature-engineered classical NLP and contextual transformer learning. Final model selection should consider F1, precision/recall trade-offs, confusion matrices, training behaviour, and computational cost.
