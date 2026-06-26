"""
app.py
-------
Main entry point for the Emotion-Based Music Recommender.
Run this file to interact with the system via the terminal.

Usage:
    python src/app.py
"""

import os
import sys
import pathlib

# Make sure imports work whether run from project root or src/
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from emotion_detector import train_and_save, load_model, predict_emotion
from music_recommender import recommend_songs, display_recommendations

MODEL_PATH = "models/emotion_model.pkl"

BANNER = """
╔══════════════════════════════════════════════════════╗
║        🎵  Emotion-Based Music Recommender  🎵        ║
║   Tell me how you feel — I'll find the perfect song  ║
╚══════════════════════════════════════════════════════╝
"""


def load_or_train_model():
    """Loads model if it exists, otherwise trains a new one."""
    pathlib.Path("models").mkdir(exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        return load_model(MODEL_PATH)
    else:
        print("No saved model found. Training a new one...")
        return train_and_save(MODEL_PATH)


def run_interactive():
    """Runs the interactive CLI loop."""
    print(BANNER)
    model = load_or_train_model()
    print("\nType how you're feeling (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Keep listening to great music 🎶")
            break

        if not user_input:
            print("Please type something about how you feel.\n")
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! Keep listening to great music 🎶")
            break

        # Detect emotion
        result = predict_emotion(user_input, model)
        emotion    = result["emotion"]
        confidence = result["confidence"]

        print(f"\nDetected emotion : {emotion.upper()} ({confidence:.0%} confident)")
        print("Score breakdown  :", end=" ")
        for e, s in list(result["all_scores"].items())[:3]:
            print(f"{e}={s:.0%}", end="  ")
        print()

        # Recommend songs
        songs = recommend_songs(emotion, top_n=5)
        display_recommendations(emotion, songs)

        print("-" * 60)
        print("Enter another feeling, or type 'quit' to exit.\n")


if __name__ == "__main__":
    run_interactive()
