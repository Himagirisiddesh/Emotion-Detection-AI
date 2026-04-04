import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import argparse

BASE_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=str(BASE_DIR / "wesad_physiological.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()

def train_bio_model():
    args = parse_args()
    
    csv_path = Path(args.data_path)
    
    # Generate mock CSV to simulate existing physiological dataset requirements
    if not csv_path.exists():
        print(f"Creating mock dataset at {csv_path} since none found...")
        np.random.seed(args.random_state)
        labels = np.random.choice([0, 1], size=200)
        hr = np.where(labels == 1, np.random.normal(95, 10, 200), np.random.normal(70, 8, 200))
        eda = np.where(labels == 1, np.random.normal(7.0, 1.5, 200), np.random.normal(3.0, 1.0, 200))
        resp = np.where(labels == 1, np.random.normal(22, 3, 200), np.random.normal(15, 2, 200))
        temp = np.where(labels == 1, np.random.normal(37.5, 0.3, 200), np.random.normal(36.5, 0.2, 200))
        
        df = pd.DataFrame({
            "heart_rate": hr,
            "eda": eda,
            "respiration": resp,
            "temperature": temp,
            "label": labels
        })
        df.to_csv(csv_path, index=False)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    original_size = len(df)
    augmented_size = 0
    
    if len(df) < 500:
        print(f"WARNING: Only {len(df)} rows found. Consider using real WESAD physiological data for better accuracy. Augmenting dataset with gaussian noise...")
        np.random.seed(args.random_state)
        augmented_dfs = [df]
        for _ in range(3):
            noisy_df = df.copy()
            for col in ['heart_rate', 'eda', 'respiration', 'temperature']:
                std_dev = noisy_df[col].std()
                noise = np.random.normal(0, 0.05, len(noisy_df)) * std_dev
                noisy_df[col] = noisy_df[col] + noise
            augmented_dfs.append(noisy_df)
        df = pd.concat(augmented_dfs, ignore_index=True)
        augmented_size = len(df)
        
    X = df[['heart_rate', 'eda', 'respiration', 'temperature']]
    y = df['label']
    
    # Pipeline
    pipeline = Pipeline([
       ("scaler", StandardScaler()),
       ("classifier", RandomForestClassifier(
           n_estimators=args.n_estimators,
           max_depth=10,
           min_samples_split=4,
           class_weight="balanced",
           random_state=args.random_state
       ))
    ])
    
    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model as: model/bio_stress_model.pkl
    model_dir = BASE_DIR / 'model'
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'bio_stress_model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"\nModel saved successfully as: {model_path}")
    print("\n--- Summary ---")
    print(f"Original dataset size: {original_size}")
    if augmented_size > 0:
        print(f"Augmented dataset size: {augmented_size}")
    print(f"CV Accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_bio_model()
