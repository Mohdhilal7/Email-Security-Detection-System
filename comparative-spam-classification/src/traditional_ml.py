import argparse
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from .config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE
from .data_utils import load_and_prepare, stratified_split
from .evaluation import evaluate_predictions, save_metrics


def train_traditional(max_samples=None):
    df, audit = load_and_prepare()
    train, valid, test = stratified_split(df)
    if max_samples:
        train = train.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), max_samples // 2), random_state=RANDOM_SEED)).reset_index(drop=True)
    vectorizer = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train["text_tfidf"])
    X_test = vectorizer.transform(test["text_tfidf"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    rows = []
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("Naive Bayes", MultinomialNB(alpha=0.5)),
    ]
    for name, model in models:
        start = time.perf_counter()
        model.fit(X_train, train["label"])
        elapsed = time.perf_counter() - start
        pred = model.predict(X_test)
        metrics = evaluate_predictions(test["label"], pred)
        metrics.update({"model": name, "training_time_seconds": elapsed})
        rows.append(metrics)
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
    save_metrics(rows, RESULTS_DIR / "metrics_traditional.csv")
    return rows, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    rows, _ = train_traditional(args.max_samples)
    print(pd.DataFrame(rows).to_string(index=False))
