import cv2
import mediapipe as mp
import time
import pyttsx3
import threading
from recognizer import predict

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Text to speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

current_word = ""
last_letter = ""
last_letter_time = 0
LETTER_DELAY = 1.5       # seconds between adding same letter
SPEAK_DELAY = 3.0        # seconds of no gesture before speaking word
last_gesture_time = time.time()
speaking = False


def speak(text):
    global speaking
    speaking = True
    engine.say(text)
    engine.runAndWait()
    speaking = False


def run():
    global current_word, last_letter, last_letter_time, last_gesture_time

    cap = cv2.VideoCapture(0)
    print("Starting real-time gesture recognition...")
    print("Show ASL hand gestures to the camera.")
    print("Press 'c' to clear word | Press 'q' to quit")

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75,
                        min_tracking_confidence=0.75) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected_letter = None

            if result.multi_hand_landmarks:
                last_gesture_time = time.time()
                for hl in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                    try:
                        letter, confidence = predict(hl)
                        if confidence > 0.6:
                            detected_letter = letter
                    except Exception as e:
                        pass

            now = time.time()

            # Add letter to word
            if detected_letter:
                if (detected_letter != last_letter or
                        now - last_letter_time > LETTER_DELAY):
                    current_word += detected_letter
                    last_letter = detected_letter
                    last_letter_time = now

            # Speak word after pause
            if (current_word and
                    now - last_gesture_time > SPEAK_DELAY and
                    not speaking):
                print(f"Speaking: {current_word}")
                t = threading.Thread(target=speak, args=(current_word,))
                t.start()
                current_word = ""
                last_letter = ""

            # Display
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"Word: {current_word}", (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            if detected_letter:
                cv2.putText(frame, f"Letter: {detected_letter}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "C=clear  Q=quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("ASL to English", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_word = ""
                last_letter = ""
                print("Word cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()