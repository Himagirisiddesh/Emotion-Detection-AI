import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from stress_text_utils import (
    BASE_DIR,
    DEFAULT_TEXT_DATASET_PATH,
    TEXT_METADATA_PATH,
    TEXT_MODEL_PATH,
    ensure_text_model_directory,
    normalize_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a binary text stress detection model from Stress.csv."
    )
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_TEXT_DATASET_PATH),
        help="Path to Stress.csv. Expected columns: text, label",
    )
    parser.add_argument(
        "--vectorizer",
        choices=["tfidf", "count"],
        default="tfidf",
        help="Text vectorizer to use.",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "naive_bayes"],
        default="logistic_regression",
        help="Classifier to train.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum vectorizer features.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency.")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency ratio.")
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3], help="Maximum n-gram size.")
    parser.add_argument("--max-iter", type=int, default=2000, help="Logistic regression max iterations.")
    return parser.parse_args()


def prepare_dataset(csv_path):
    dataframe = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    dataframe = dataframe[["text", "label"]].copy()
    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="coerce")
    dataframe = dataframe.dropna(subset=["label"]).copy()
    dataframe["label"] = dataframe["label"].astype(int)

    invalid_labels = sorted(set(dataframe["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Unexpected labels found in {csv_path}: {invalid_labels}. Expected only 0 and 1.")

    dataframe = dataframe.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return dataframe


def build_vectorizer(args):
    vectorizer_kwargs = {
        "preprocessor": normalize_text,
        "lowercase": False,
        "ngram_range": (1, args.ngram_max),
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
    }

    if args.vectorizer == "count":
        return CountVectorizer(**vectorizer_kwargs)

    return TfidfVectorizer(sublinear_tf=True, **vectorizer_kwargs)


def build_classifier(args):
    if args.model == "naive_bayes":
        return MultinomialNB()

    return LogisticRegression(
        max_iter=args.max_iter,
        solver="liblinear",
        class_weight="balanced",
        random_state=args.random_state,
    )


def build_pipeline(args):
    return Pipeline(
        steps=[
            ("vectorizer", build_vectorizer(args)),
            ("classifier", build_classifier(args)),
        ]
    )


def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Not Stressed", "Stressed"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "classification_report": report,
    }


def save_pipeline(pipeline, model_path):
    with Path(model_path).open("wb") as file:
        pickle.dump(pipeline, file)


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    ensure_text_model_directory()
    dataframe = prepare_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=dataframe["label"],
    )

    pipeline = build_pipeline(args)

    print("Training text stress model...")
    print(f"Dataset path: {dataset_path}")
    print(f"Total rows: {len(dataframe)}")
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Label distribution: {dataframe['label'].value_counts().sort_index().to_dict()}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    metrics = evaluate_predictions(y_test, y_pred)

    save_pipeline(pipeline, TEXT_MODEL_PATH)

    metadata = {
        "labels": [0, 1],
        "label_names": {"0": "Not Stressed", "1": "Stressed"},
        "dataset_path": str(dataset_path),
        "model_path": str(TEXT_MODEL_PATH.relative_to(BASE_DIR)),
        "vectorizer": type(pipeline.named_steps["vectorizer"]).__name__,
        "classifier": type(pipeline.named_steps["classifier"]).__name__,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "total_rows": int(len(dataframe)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "ngram_range": [1, args.ngram_max],
        "decision_threshold": 0.55,
        "confidence_threshold": 0.65,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with TEXT_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nEvaluation Metrics")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"\nSaved model to   : {TEXT_MODEL_PATH}")
    print(f"Saved metadata to: {TEXT_METADATA_PATH}")


if __name__ == "__main__":
    main()
