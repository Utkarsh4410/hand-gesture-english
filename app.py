import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import threading
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from features import extract_features_from_landmarks

# Load model
@st.cache_resource
def load_model():
    with open("models/gesture_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def predict_gesture(hand_landmarks):
    features = extract_features_from_landmarks(hand_landmarks).reshape(1, -1)
    letter = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return letter, confidence

# UI
st.set_page_config(page_title="ASL to English", page_icon="🤟", layout="centered")
st.title("🤟 Hand Gesture to English Translator")
st.caption("Show ASL hand gestures to your webcam — letters will appear below!")


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.buffer = deque(maxlen=10)
        self.last_added_time = 0
        self.last_letter = ""
        self.detected_letter = ""
        self.confidence = 0
        self._current_word = ""
        self._lock = threading.Lock()

    def get_smoothed(self):
        if len(self.buffer) < 5:
            return None, 0
        letters = [p[0] for p in self.buffer]
        confs = [p[1] for p in self.buffer]
        most_common, count = Counter(letters).most_common(1)[0]
        consistency = count / len(letters)
        avg_conf = sum(confs) / len(confs)
        if consistency >= 0.6 and avg_conf >= 0.65:
            return most_common, avg_conf
        return None, 0

    def get_word(self):
        with self._lock:
            return self._current_word

    def clear_word(self):
        with self._lock:
            self._current_word = ""
            self.last_letter = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        now = time.time()
        self.detected_letter = ""

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
                try:
                    letter, conf = predict_gesture(hl)
                    self.buffer.append((letter, conf))
                except:
                    pass

            smoothed, conf = self.get_smoothed()
            if smoothed:
                self.detected_letter = smoothed
                self.confidence = conf
                with self._lock:
                    if (smoothed != self.last_letter or
                            now - self.last_added_time > 2.0):
                        self._current_word += smoothed
                        self.last_letter = smoothed
                        self.last_added_time = now
        else:
            self.buffer.clear()

        # Draw on frame
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, h-70), (w, h), (0, 0, 0), -1)
        with self._lock:
            word = self._current_word
        cv2.putText(img, f"Word: {word}", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if self.detected_letter:
            cv2.putText(img, f"{self.detected_letter} ({self.confidence*100:.0f}%)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        elif result.multi_hand_landmarks:
            cv2.putText(img, "Hold steady...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        else:
            cv2.putText(img, "No hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="gesture",
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

col1, col2 = st.columns([2, 1])
with col2:
    if st.button("🗑️ Clear Word"):
        if ctx.video_processor:
            ctx.video_processor.clear_word()

st.markdown("---")

# Display current word — read from processor (thread-safe)
word_placeholder = st.empty()
if ctx.state.playing and ctx.video_processor:
    word = ctx.video_processor.get_word()
    word_placeholder.markdown(f"### 📝 Current Word: `{word}`")
else:
    word_placeholder.markdown("### 📝 Current Word: `` ")

st.markdown("**Tips:** Hold each gesture steady for 1-2 seconds • Good lighting helps • Keep hand fully in frame")
st.markdown("*The recognized word is also displayed live on the video overlay at the bottom of the frame.*")