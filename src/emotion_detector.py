"""
emotion_detector.py
--------------------
Detects the emotion from a piece of text using a simple but effective
TF-IDF + Logistic Regression pipeline.

Emotions supported: happy, sad, angry, fearful, surprised, calm
"""

import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 1.  Training data  (small built-in dataset — expandable)
# ---------------------------------------------------------------------------

TRAINING_DATA = [
    ("I feel amazing today, everything is going great!", "happy"),
    ("I am so excited and full of joy right now!", "happy"),
    ("Life is beautiful, I love every moment of it.", "happy"),
    ("I just got promoted! I am thrilled and overjoyed!", "happy"),
    ("Today was perfect, I feel blessed and grateful.", "happy"),
    ("I am laughing and smiling all day long!", "happy"),
    ("Everything feels wonderful and full of sunshine.", "happy"),
    ("I am so happy I could dance all night.", "happy"),
    ("Great news came in today, feeling on top of the world!", "happy"),
    ("Pure bliss, I am living my best life.", "happy"),
    ("I feel so lonely and empty inside.", "sad"),
    ("Nothing seems to go right, I am so down.", "sad"),
    ("I miss them so much, my heart aches.", "sad"),
    ("I cried myself to sleep last night.", "sad"),
    ("Everything feels hopeless and grey today.", "sad"),
    ("I feel broken and lost without direction.", "sad"),
    ("The grief is overwhelming, I can barely breathe.", "sad"),
    ("I am so disappointed, all my hopes are crushed.", "sad"),
    ("Feeling deeply melancholic and nostalgic.", "sad"),
    ("I just want to disappear and be alone.", "sad"),
    ("I am so frustrated and furious right now!", "angry"),
    ("This is absolutely infuriating, I cannot stand it!", "angry"),
    ("I am boiling with rage, how dare they!", "angry"),
    ("Everything makes me so mad today.", "angry"),
    ("I am outraged by what happened, it is unacceptable.", "angry"),
    ("Stop pushing me, I am at my breaking point!", "angry"),
    ("I feel intense anger and resentment.", "angry"),
    ("That was so unfair, I am livid!", "angry"),
    ("I hate this situation, it drives me crazy.", "angry"),
    ("I can feel my blood pressure rising with anger.", "angry"),
    ("I am so scared and anxious about the future.", "fearful"),
    ("My heart is pounding, I feel terrified.", "fearful"),
    ("I keep having nightmares, something feels very wrong.", "fearful"),
    ("I am paralysed with fear, unable to move forward.", "fearful"),
    ("The uncertainty is driving me into a panic.", "fearful"),
    ("I feel dread every time I think about it.", "fearful"),
    ("My anxiety is through the roof, I am trembling.", "fearful"),
    ("I am frightened by what might happen next.", "fearful"),
    ("The fear of failure keeps me up at night.", "fearful"),
    ("I feel vulnerable and unsafe right now.", "fearful"),
    ("I cannot believe what just happened, I am shocked!", "surprised"),
    ("That came completely out of nowhere, wow!", "surprised"),
    ("I am totally astonished, I never expected this.", "surprised"),
    ("What a twist! I am completely blown away.", "surprised"),
    ("I am stunned, this is absolutely unbelievable!", "surprised"),
    ("My jaw dropped, I did not see that coming.", "surprised"),
    ("That news was the biggest surprise of my life.", "surprised"),
    ("I am pleasantly surprised by this outcome.", "surprised"),
    ("Everything happened so fast, I am still in shock.", "surprised"),
    ("I never imagined this would happen, incredible!", "surprised"),
    ("I feel peaceful and completely at ease.", "calm"),
    ("Everything is quiet, I am totally relaxed.", "calm"),
    ("I am in a state of serene contentment.", "calm"),
    ("Nothing bothers me today, I feel centred.", "calm"),
    ("The stillness around me brings deep tranquillity.", "calm"),
    ("I feel balanced and grounded in this moment.", "calm"),
    ("Breathing slowly, I am at total peace with myself.", "calm"),
    ("Gentle and steady, that is how I feel right now.", "calm"),
    ("I am meditating and feeling wonderfully calm.", "calm"),
    ("The world feels gentle and unhurried today.", "calm"),
]


# ---------------------------------------------------------------------------
# 2.  Text pre-processing helper
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3.  Model builder
# ---------------------------------------------------------------------------

def build_model() -> Pipeline:
    """
    Builds a simple sklearn Pipeline:
        TF-IDF vectoriser  ->  Logistic Regression classifier
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=5000,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
        )),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# 4.  Train & save
# ---------------------------------------------------------------------------

def train_and_save(save_path: str = "models/emotion_model.pkl") -> Pipeline:
    """Trains the model on TRAINING_DATA and saves it with pickle."""
    import pathlib
    pathlib.Path("models").mkdir(exist_ok=True)

    texts, labels = zip(*TRAINING_DATA)
    model = build_model()
    model.fit(texts, labels)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained on {len(texts)} samples and saved to '{save_path}'")
    return model


# ---------------------------------------------------------------------------
# 5.  Load & predict
# ---------------------------------------------------------------------------

def load_model(model_path: str = "models/emotion_model.pkl") -> Pipeline:
    """Loads a previously saved model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_emotion(text: str, model: Pipeline) -> dict:
    """
    Returns:
        {
            'emotion'    : str   - top predicted emotion,
            'confidence' : float - probability of top emotion (0-1),
            'all_scores' : dict  - probabilities for every emotion,
        }
    """
    classes = model.classes_
    proba   = model.predict_proba([text])[0]

    scores      = dict(zip(classes, proba.tolist()))
    top_emotion = max(scores, key=scores.get)
    confidence  = scores[top_emotion]

    return {
        "emotion"   : top_emotion,
        "confidence": round(confidence, 4),
        "all_scores": {k: round(v, 4) for k, v in sorted(scores.items(),
                                                          key=lambda x: -x[1])},
    }


# ---------------------------------------------------------------------------
# 6.  Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = train_and_save()

    test_sentences = [
        "I feel absolutely fantastic today!",
        "I am terrified of what might happen.",
        "This whole situation makes me so angry.",
        "I just want some peace and quiet.",
        "I cannot believe that just happened!",
        "My heart is heavy and I feel like crying.",
    ]

    print("\n--- Emotion Predictions ---")
    for sent in test_sentences:
        result = predict_emotion(sent, model)
        print(f"  '{sent}'")
        print(f"   -> {result['emotion'].upper()}  (confidence: {result['confidence']:.0%})\n")
