import os
import csv
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = "data"
MODEL_PATH = "models/gesture_model.pkl"

os.makedirs("models", exist_ok=True)


def load_data():
    X, y = [], []
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
                    X.append([float(v) for v in row])
                    y.append(letter)

    return np.array(X), np.array(y)


def train():
    print("Loading data...")
    X, y = load_data()

    if len(X) == 0:
        print("No data found! Run collector.py first.")
        return

    print(f"Total samples: {len(X)} across {len(set(y))} letters")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()