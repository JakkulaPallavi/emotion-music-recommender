# 🎵 Emotion-Based Music Recommender

> **An AI/ML system that reads how you feel — and finds the perfect song for it.**

[![CI](https://github.com/YOUR_USERNAME/emotion-music-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/emotion-music-recommender/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📖 What Is This?

You type how you're feeling — in plain English, like a journal entry — and the system:

1. **Detects your emotion** using a trained NLP classifier (TF-IDF + Logistic Regression)
2. **Recommends songs** from a curated database that match your mood
3. **Links you to Spotify** so you can start listening instantly

It supports **6 emotions**: `happy` 😊 · `sad` 😢 · `angry` 😠 · `fearful` 😨 · `surprised` 😲 · `calm` 😌

---

## 🖥️ Demo

```
╔══════════════════════════════════════════════════════╗
║        🎵  Emotion-Based Music Recommender  🎵        ║
║   Tell me how you feel — I'll find the perfect song  ║
╚══════════════════════════════════════════════════════╝

You: I've been feeling really anxious about my exams, like everything is crashing down

Detected emotion : FEARFUL (74% confident)
Score breakdown  : fearful=74%  sad=12%  angry=8%

😨  You're feeling FEARFUL — here are your songs:

1. Fear  —  Kendrick Lamar
   Genre : Hip-Hop
   Vibe  : anxiety, introspective, raw
   Listen: https://open.spotify.com/search/Fear%20Kendrick%20Lamar

2. Safe and Sound  —  Taylor Swift ft. Civil Wars
   Genre : Folk Pop
   Vibe  : comfort, reassurance, soft
   Listen: https://open.spotify.com/search/Safe%20and%20Sound%20Taylor%20Swift
...
```

---

## 🧠 How It Works

```
Your Text Input
      │
      ▼
┌─────────────────────────────────────┐
│  Text Pre-processing                │
│  • Lowercase                        │
│  • Remove punctuation               │
│  • Normalise whitespace             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  TF-IDF Vectoriser                  │
│  • Converts words to weighted nums  │
│  • Captures single & 2-word phrases │
│  • Vocabulary of up to 5,000 terms  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Logistic Regression Classifier     │
│  • Trained on 60 labelled samples   │
│  • Outputs probability per emotion  │
│  • Saved as models/emotion_model.pkl│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Music Recommendation Engine        │
│  • Looks up emotion in song DB      │
│  • Returns top-N matching songs     │
│  • Generates Spotify search URLs    │
└─────────────────────────────────────┘
               │
               ▼
       🎵 Song Recommendations
```

### Why TF-IDF + Logistic Regression?

| Reason | Detail |
|--------|--------|
| **Beginner-friendly** | No GPU required, runs on any laptop |
| **Interpretable** | You can see exactly which words drive predictions |
| **Fast** | Trains in under a second, predicts in milliseconds |
| **Extendable** | Easy to swap in a neural model later |

---

## 📁 Project Structure

```
emotion-music-recommender/
│
├── src/
│   ├── emotion_detector.py     # NLP model: training, saving, predicting
│   ├── music_recommender.py    # Song database + recommendation logic
│   ├── app.py                  # Interactive CLI entry point
│   └── evaluate.py             # Cross-validation & metrics report
│
├── tests/
│   ├── test_emotion_detector.py   # 15 unit tests for the ML pipeline
│   └── test_music_recommender.py  # 11 unit tests for recommendation logic
│
├── notebooks/
│   └── exploration.py          # Visual exploration of model predictions
│
├── models/
│   └── emotion_model.pkl       # Saved trained model (auto-generated)
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: auto-test on every push
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/emotion-music-recommender.git
cd emotion-music-recommender
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python src/app.py
```

The model will train automatically on first run (~1 second) and save itself to `models/emotion_model.pkl`. Subsequent runs will load the saved model instantly.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_emotion_detector.py::TestCleanText::test_lowercase          PASSED
tests/test_emotion_detector.py::TestModel::test_model_has_six_classes  PASSED
...
======================== 26 passed in 1.4s ========================
```

---

## 📊 Model Evaluation

Run the evaluation script to see cross-validation scores and a full classification report:

```bash
python src/evaluate.py
```

```
Cross-Validation Accuracy (5-fold):
  Mean : 20.00%  ← low because dataset is tiny (60 samples)
  
Classification Report (trained on full set):
  accuracy : 1.00  ← perfect fit on training data (expected)
```

> **Note:** The cross-validation score is low because the training set is intentionally small (60 samples). To improve real-world performance, add more labelled examples to `TRAINING_DATA` in `emotion_detector.py`. With 500+ samples per class, accuracy typically reaches 80-90%.

---

## 🔧 How to Extend

### Add more training data

Open `src/emotion_detector.py` and add tuples to `TRAINING_DATA`:

```python
TRAINING_DATA = [
    ...
    ("Your new sentence here", "happy"),   # add as many as you like
    ("Another example of sadness",  "sad"),
]
```

### Add more songs

Open `src/music_recommender.py` and add dicts to `SONG_DATABASE`:

```python
SONG_DATABASE = {
    "happy": [
        ...
        {
            "title": "Your Song Title",
            "artist": "Artist Name",
            "genre": "Pop",
            "mood_tags": ["upbeat", "energetic"],
        },
    ],
}
```

### Swap in a better model

Replace the `build_model()` function in `emotion_detector.py` with any sklearn-compatible estimator — e.g. `RandomForestClassifier`, `SVC`, or even a PyTorch/HuggingFace model wrapped in a sklearn interface.

---

## 🗺️ Roadmap

- [ ] Web UI using Flask or Streamlit
- [ ] Spotify API integration for real track metadata & previews
- [ ] Support for more emotions (disgust, anticipation, trust)
- [ ] Multi-language support
- [ ] Fine-tuned BERT model for better accuracy on short texts
- [ ] Dataset export/import (CSV support)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built by **Your Name** — feel free to connect on [LinkedIn](https://linkedin.com) or [Twitter](https://twitter.com).

---

*Built with using Python, scikit-learn, and a passion for music.*
