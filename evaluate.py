import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from stress_text_utils import TextStressPredictor, FACE_STRESS_SCORES, combine_stress_assessment

def load_bio_model(path):
    model_path = Path(path)
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

def evaluate_text_only(df, predictor):
    y_true = df["true_label"].values
    y_pred = []
    for text in df["text"]:
        pred = predictor.predict(text)
        y_pred.append(1 if pred["stress_level"] == "HIGH" else 0)
    return compute_metrics(y_true, y_pred)

def evaluate_face_only(df):
    y_true = df["true_label"].values
    y_pred = []
    for label in df["face_label"]:
        score = FACE_STRESS_SCORES.get(label, 1)
        y_pred.append(1 if score >= 1 else 0)
    return compute_metrics(y_true, y_pred)

def evaluate_bio_only(df, bio_model):
    features = df[["heart_rate", "eda", "respiration", "temperature"]].values
    y_pred = bio_model.predict(features)
    y_true = df["true_label"].values
    return compute_metrics(y_true, y_pred)

def evaluate_multimodal(df, predictor, bio_model):
    y_true = df["true_label"].values
    y_pred = []
    bio_predictions = bio_model.predict(df[["heart_rate", "eda", "respiration", "temperature"]].values)
    bio_probs = bio_model.predict_proba(df[["heart_rate", "eda", "respiration", "temperature"]].values)
    
    for idx, row in df.iterrows():
        face_pred = {"label": row["face_label"], "confidence": row["face_confidence"]}
        text_pred = predictor.predict(row["text"])
        
        bio_pred_val = bio_predictions[idx]
        bio_prob = max(bio_probs[idx])
        bio_pred_dict = {
            "status": "Stressed" if bio_pred_val == 1 else "Not Stressed",
            "confidence": bio_prob
        }
        
        result = combine_stress_assessment(face_pred, text_pred, bio_pred_dict)
        y_pred.append(1 if result["score"] >= 1.0 else 0)
        
    return compute_metrics(y_true, y_pred)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def print_report(name, metrics):
    print(f"│ {name:<15} │ {metrics['Accuracy']:8.4f} │ {metrics['Precision']:9.4f} │ {metrics['Recall']:6.4f} │ {metrics['F1']:6.4f} │")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--bio-model", default="model/bio_stress_model.pkl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.test_csv)
    predictor = TextStressPredictor()
    bio_model = load_bio_model(args.bio_model)
    
    metrics_text = evaluate_text_only(df, predictor)
    metrics_face = evaluate_face_only(df)
    metrics_bio = evaluate_bio_only(df, bio_model)
    metrics_multi = evaluate_multimodal(df, predictor, bio_model)
    
    print("┌─────────────────┬──────────┬───────────┬────────┬────────┐")
    print("│ Modality        │ Accuracy │ Precision │ Recall │   F1   │")
    print("├─────────────────┼──────────┼───────────┼────────┼────────┤")
    print_report("Text Only", metrics_text)
    print_report("Face Only", metrics_face)
    print_report("Bio Only", metrics_bio)
    print_report("Multimodal", metrics_multi)
    print("└─────────────────┴──────────┴───────────┴────────┴────────┘")
    
    single_mods = {"Text": metrics_text, "Face": metrics_face, "Bio": metrics_bio}
    best_single_name = max(single_mods, key=lambda k: single_mods[k]["F1"])
    best_single_f1 = single_mods[best_single_name]["F1"]
    multi_f1 = metrics_multi["F1"]
    
    # Avoid div zero
    if best_single_f1 > 0:
        improvement = ((multi_f1 - best_single_f1) / best_single_f1) * 100
    else:
        improvement = 0.0
        
    print(f"Best single modality: {best_single_name} with F1={best_single_f1:.4f}. Multimodal F1={multi_f1:.4f}. Improvement: +{improvement:.2f}%")
    
    if args.output:
        out_df = pd.DataFrame([
            {"Modality": "Text Only", **metrics_text},
            {"Modality": "Face Only", **metrics_face},
            {"Modality": "Bio Only", **metrics_bio},
            {"Modality": "Multimodal", **metrics_multi}
        ])
        out_df.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
