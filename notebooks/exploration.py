"""
exploration.py
---------------
A standalone script that mirrors what you'd do in a Jupyter notebook:
visualise emotion score distributions and test edge cases.

Run with:
    python notebooks/exploration.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from emotion_detector import train_and_save, predict_emotion
from music_recommender import recommend_songs

pathlib.Path("models").mkdir(exist_ok=True)
model = train_and_save("models/emotion_model.pkl")

samples = [
    "I woke up feeling incredible and ready to take on the world.",
    "The silence in this empty house is deafening.",
    "I want to scream at the top of my lungs.",
    "My palms are sweaty and I cannot stop shaking.",
    "Wait, did that really just happen?",
    "I could sit here by the lake all afternoon.",
    "Everything is fine but something feels slightly off.",
    "This is the worst day of my entire life.",
    "I am nervous but also weirdly excited.",
    "Just breathing and being. Nothing else.",
]

print("\n" + "=" * 65)
print("  EMOTION SCORE EXPLORER")
print("=" * 65)

for text in samples:
    result = predict_emotion(text, model)
    e = result["emotion"]
    c = result["confidence"]
    bar = "X" * int(c * 20) + "-" * (20 - int(c * 20))
    print(f"\n  Text    : {text[:55]}")
    print(f"  Emotion : {e.upper():<12}  [{bar}] {c:.0%}")
    scores_str = "  ".join(f"{em}={sc:.0%}" for em, sc in result["all_scores"].items())
    print(f"  Scores  : {scores_str}")

print("\n" + "=" * 65)
print("  SAMPLE RECOMMENDATIONS")
print("=" * 65)
for em in ["happy", "sad", "calm"]:
    songs = recommend_songs(em, top_n=2)
    print(f"\n  {em.upper()} picks:")
    for s in songs:
        print(f"    - {s['title']} by {s['artist']} [{s['genre']}]")
