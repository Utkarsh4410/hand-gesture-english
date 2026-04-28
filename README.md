# 🤟 Hand Gesture to English Translator

A real-time **ASL (American Sign Language) hand gesture recognition** system that translates hand signs into English letters using computer vision and machine learning.

Built with **MediaPipe** for hand tracking, **scikit-learn** for gesture classification, and **Streamlit** for a web-based interface.

---

## ✨ Features

- 🖐️ **Real-time hand tracking** using MediaPipe's 21-point hand landmark detection
- 🔤 **ASL alphabet recognition** (A–Z) with confidence scoring
- 🧠 **Engineered feature extraction** — 86 features including normalized coordinates, finger curl ratios, inter-finger angles, and fingertip distances
- 🌐 **Streamlit web app** with live webcam feed via WebRTC
- 🖥️ **Desktop mode** with OpenCV window + text-to-speech output
- 📊 **Smart prediction smoothing** using a rolling buffer for stable results

---

## 🏗️ Project Architecture

```
hand-gesture-english/
├── app.py            # Streamlit web app (WebRTC-based live demo)
├── main.py           # Desktop version (OpenCV + text-to-speech)
├── collector.py      # Data collection tool (record ASL gestures)
├── trainer.py        # Model training pipeline
├── features.py       # Shared feature engineering (86 features)
├── recognizer.py     # Prediction module
├── requirements.txt  # Python dependencies
├── data/             # Collected gesture data (CSV per letter)
└── models/           # Trained model (gesture_model.pkl)
```

---

## 🔧 How It Works

```
Webcam → MediaPipe Hand Detection → 21 Landmarks → Feature Engineering (86 features) → Random Forest Classifier → Predicted Letter
```

### Feature Engineering Pipeline

The system extracts **86 engineered features** from 21 hand landmarks:

| Feature Type                      | Count | Description                                  |
|-----------------------------------|-------|----------------------------------------------|
| Normalized coordinates            | 63    | Wrist-centered, palm-size-scaled (x, y, z)   |
| Fingertip-to-wrist distances      | 5     | How far each finger extends from wrist        |
| Adjacent fingertip distances      | 4     | Spread between neighboring fingers            |
| Fingertip-to-palm distances       | 5     | Distance from each tip to palm center         |
| Finger curl ratios                | 5     | How bent each finger is (0 = curled, 1 = straight) |
| Inter-finger angles               | 4     | Angle between adjacent finger directions      |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/Utkarsh4410/hand-gesture-english.git
cd hand-gesture-english

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional dependencies for the web app

```bash
pip install streamlit streamlit-webrtc
```

---

## 📖 Usage

### Step 1: Collect Training Data

Record hand gesture samples for each letter (A–Z):

```bash
python collector.py
```

- Show the gesture for the prompted letter
- Press **SPACE** to start collecting (300 samples per letter)
- Press **Q** to quit

### Step 2: Train the Model

```bash
python trainer.py
```

This trains a **Random Forest classifier** (300 trees) with feature scaling and outputs:
- Test accuracy
- 5-fold cross-validation score
- Worst-performing letters

### Step 3: Run the App

**Web App (Streamlit):**
```bash
streamlit run app.py
```

**Desktop App (OpenCV):**
```bash
python main.py
```

| Control       | Action                     |
|---------------|----------------------------|
| `C` key       | Clear current word          |
| `Q` key       | Quit application            |
| 3s pause      | Auto-speak the formed word  |

---

## 🛠️ Tech Stack

| Technology    | Purpose                        |
|---------------|--------------------------------|
| MediaPipe     | Hand landmark detection        |
| OpenCV        | Video capture & processing     |
| scikit-learn  | ML model (Random Forest)       |
| NumPy         | Feature computation            |
| Streamlit     | Web interface                  |
| WebRTC        | Browser webcam streaming       |
| pyttsx3       | Text-to-speech (desktop mode)  |

---

## 📈 Model Performance

The model uses a `Pipeline` with `StandardScaler` + `RandomForestClassifier` (300 estimators) for robust predictions. Key design choices:

- **Prediction smoothing**: Rolling buffer of 10 frames with consistency threshold (60%) to avoid flickering
- **Confidence filtering**: Only accepts predictions above 65% average confidence
- **Duplicate prevention**: Same letter won't repeat unless held for 2+ seconds

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Utkarsh** — [@Utkarsh4410](https://github.com/Utkarsh4410)

---

> 💡 **Tip:** For best results, ensure good lighting and keep your hand fully visible in the camera frame.
