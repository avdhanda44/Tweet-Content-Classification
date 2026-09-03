"""Train comparable text classifiers and save metrics and the best pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "tweets_cleaned.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_PATH = ROOT / "reports" / "model_results.csv"
LABEL_NAMES = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}


def main() -> None:
    data = pd.read_csv(DATA_PATH).dropna(subset=["clean_tweet", "class"])
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_tweet"], data["class"].astype(int), test_size=0.2,
        random_state=42, stratify=data["class"],
    )
    models = {
        "BoW + Logistic Regression": Pipeline([
            ("features", CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]),
        "TF-IDF + Logistic Regression": Pipeline([
            ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        ]),
        "TF-IDF + Linear SVM": Pipeline([
            ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
    }
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        rows.append({"model": name, "accuracy": report["accuracy"], "macro_f1": f1_score(y_test, predictions, average="macro")})
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name = results.iloc[0]["model"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.to_string(index=False))
    print(f"Saved best model: {MODEL_PATH}")
    print(f"Saved metrics: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
