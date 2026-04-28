import os
import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from features import extract_features_from_raw

DATA_DIR = "data"
MODEL_PATH = "models/gesture_model.pkl"

os.makedirs("models", exist_ok=True)


def load_data():
    X_raw, y = [], []
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    for letter in letters:
        filepath = os.path.join(DATA_DIR, f"{letter}.csv")
        if not os.path.exists(filepath):
            print(f"[SKIP] No data for {letter}")
            continue

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    X_raw.append([float(v) for v in row])
                    y.append(letter)

    return X_raw, np.array(y)


def train():
    print("Loading data...")
    X_raw, y = load_data()

    if len(X_raw) == 0:
        print("No data found! Run collector.py first.")
        return

    print(f"Total samples: {len(X_raw)} across {len(set(y))} letters")

    # Extract engineered features
    print("Extracting engineered features...")
    X = np.array([extract_features_from_raw(row) for row in X_raw])
    print(f"Feature vector size: {X.shape[1]} (was 63 raw)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline with scaling + classifier
    print("Training model (RandomForest with scaling)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Cross-validation score
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # Per-letter breakdown (show worst performing)
    print("\nPer-letter accuracy:")
    report = classification_report(y_test, y_pred, output_dict=True)
    worst = sorted(
        [(k, v['f1-score']) for k, v in report.items() if len(k) == 1],
        key=lambda x: x[1]
    )[:5]
    print("  Worst 5 letters:")
    for letter, f1 in worst:
        print(f"    {letter}: {f1 * 100:.1f}% F1")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()