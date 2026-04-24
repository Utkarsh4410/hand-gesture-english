import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "data"
SAMPLES_PER_LETTER = 300
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

os.makedirs(DATA_DIR, exist_ok=True)

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def collect():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        for letter in LETTERS:
            filepath = os.path.join(DATA_DIR, f"{letter}.csv")
            if os.path.exists(filepath):
                print(f"[SKIP] {letter} already collected.")
                continue

            print(f"\n>>> Get ready to show gesture for: {letter}")
            print("Press SPACE to start collecting...")

            samples = []
            collecting = False
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for hl in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                status = f"Letter: {letter} | Collected: {count}/{SAMPLES_PER_LETTER}"
                if not collecting:
                    status += " | Press SPACE to start"
                cv2.putText(frame, status, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not collecting:
                    collecting = True
                    print(f"Collecting {letter}...")
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if collecting and result.multi_hand_landmarks:
                    landmarks = extract_landmarks(result.multi_hand_landmarks[0])
                    samples.append(landmarks)
                    count += 1
                    if count >= SAMPLES_PER_LETTER:
                        break

            # Save to CSV
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                for row in samples:
                    writer.writerow(row)
            print(f"[SAVED] {letter}.csv with {count} samples")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")


if __name__ == "__main__":
    collect()