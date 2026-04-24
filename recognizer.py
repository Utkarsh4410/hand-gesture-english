import pickle
import numpy as np

MODEL_PATH = "models/gesture_model.pkl"

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def predict(hand_landmarks):
    """
    Takes a mediapipe hand_landmarks object and returns predicted letter.
    """
    model = load_model()
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    features = np.array(coords).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return prediction, confidence