import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, TEST_SIZE, VALID_SIZE

TEXT_CANDIDATES = ["text", "email_text", "message", "body", "email", "content", "mail", "email body", "email_body"]
LABEL_CANDIDATES = ["label", "spam", "category", "class", "target", "type", "spam_or_not"]


def find_csv(data_dir: Path = DATA_DIR) -> Path:
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {data_dir}. Download the required Kaggle dataset and place it under data/raw/.")
    if len(csvs) == 1:
        return csvs[0]
    for path in csvs:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        if {str(c).strip().lower() for c in cols}.intersection({c.lower() for c in LABEL_CANDIDATES}):
            return path
    return csvs[0]


def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    text_col = next((lower_map[c] for c in TEXT_CANDIDATES if c in lower_map), None)
    label_col = next((lower_map[c] for c in LABEL_CANDIDATES if c in lower_map), None)
    if text_col is None:
        object_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(object_cols):
            text_col = max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if label_col is None:
        candidates = [c for c in df.columns if df[c].nunique(dropna=True) <= 10]
        if candidates:
            label_col = candidates[-1]
    if text_col is None or label_col is None:
        raise ValueError(f"Could not infer text/label columns. Available columns: {list(df.columns)}")
    return text_col, label_col


def normalize_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"spam", "1", "true", "yes", "junk"}:
        return 1
    if s in {"ham", "0", "false", "no", "not spam", "legitimate", "normal"}:
        return 0
    try:
        number = float(s)
        if number in (0.0, 1.0):
            return int(number)
    except ValueError:
        pass
    return None


def clean_for_tfidf(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " EMAILTOKEN ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9_!?$%\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_for_transformer(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_and_prepare():
    path = find_csv()
    df = pd.read_csv(path)
    text_col, label_col = infer_columns(df)
    work = df[[text_col, label_col]].copy()
    before = len(work)
    work[text_col] = work[text_col].fillna("").astype(str).str.strip()
    work["label"] = work[label_col].map(normalize_label)
    missing_text = int((work[text_col] == "").sum())
    invalid_labels = int(work["label"].isna().sum())
    work = work[(work[text_col] != "") & work["label"].notna()].copy()
    work["label"] = work["label"].astype(int)
    duplicate_count = int(work.duplicated(subset=[text_col, "label"]).sum())
    work = work.drop_duplicates(subset=[text_col, "label"]).reset_index(drop=True)
    work["text_tfidf"] = work[text_col].map(clean_for_tfidf)
    work["text_transformer"] = work[text_col].map(clean_for_transformer)
    audit = {
        "source_file": str(path), "original_rows": before, "rows_after_cleaning": int(len(work)),
        "text_column": str(text_col), "label_column": str(label_col),
        "empty_text_rows_removed": missing_text, "invalid_labels_removed": invalid_labels,
        "duplicate_text_label_rows_removed": duplicate_count,
        "class_distribution": {str(k): int(v) for k, v in work["label"].value_counts().sort_index().items()},
        "text_length_mean": float(work[text_col].str.len().mean()),
        "text_length_median": float(work[text_col].str.len().median()),
        "text_length_max": int(work[text_col].str.len().max()),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return work, audit


def stratified_split(df):
    train_val, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df["label"])
    relative_valid = VALID_SIZE / (1 - TEST_SIZE)
    train, valid = train_test_split(train_val, test_size=relative_valid, random_state=RANDOM_SEED, stratify=train_val["label"])
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    df, audit = load_and_prepare()
    train, valid, test = stratified_split(df)
    print(json.dumps(audit, indent=2))
    print(f"Split sizes: train={len(train)}, validation={len(valid)}, test={len(test)}")
