"""
test_emotion_detector.py
-------------------------
Unit tests for the emotion detection module.
Run with:   pytest tests/ -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
from emotion_detector import clean_text, build_model, train_and_save, predict_emotion, TRAINING_DATA


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        assert clean_text("Hello, World!") == "hello world"

    def test_strips_whitespace(self):
        assert clean_text("  spaces  ") == "spaces"

    def test_empty_string(self):
        assert clean_text("") == ""


class TestModel:
    @pytest.fixture(scope="class")
    def trained_model(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("models")
        model_path = str(tmp / "test_model.pkl")
        return train_and_save(save_path=model_path)

    def test_model_has_six_classes(self, trained_model):
        expected = {"happy", "sad", "angry", "fearful", "surprised", "calm"}
        assert set(trained_model.classes_) == expected

    def test_predict_returns_dict(self, trained_model):
        result = predict_emotion("I feel great!", trained_model)
        assert "emotion" in result and "confidence" in result and "all_scores" in result

    def test_confidence_between_0_and_1(self, trained_model):
        result = predict_emotion("I am so happy today!", trained_model)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_scores_sum_to_one(self, trained_model):
        result = predict_emotion("This makes me furious.", trained_model)
        total = sum(result["all_scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_happy_emotion_detected(self, trained_model):
        result = predict_emotion("I feel amazing, joyful and thrilled today!", trained_model)
        assert result["emotion"] == "happy", f"Expected happy but got {result['emotion']}"

    def test_sad_emotion_detected(self, trained_model):
        assert predict_emotion("I feel so lonely and heartbroken.", trained_model)["emotion"] == "sad"

    def test_angry_emotion_detected(self, trained_model):
        assert predict_emotion("I am furious and filled with rage!", trained_model)["emotion"] == "angry"

    def test_calm_emotion_detected(self, trained_model):
        assert predict_emotion("I feel peaceful, relaxed and at ease.", trained_model)["emotion"] == "calm"


class TestTrainingData:
    def test_has_enough_samples(self):
        assert len(TRAINING_DATA) >= 60

    def test_all_entries_are_tuples(self):
        for entry in TRAINING_DATA:
            assert isinstance(entry, tuple) and len(entry) == 2

    def test_six_emotion_classes(self):
        labels = [label for _, label in TRAINING_DATA]
        assert len(set(labels)) == 6
