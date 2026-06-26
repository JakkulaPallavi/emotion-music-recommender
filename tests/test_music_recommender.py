"""
test_music_recommender.py
--------------------------
Unit tests for the music recommendation module.
Run with:   pytest tests/ -v
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pytest
from music_recommender import recommend_songs, get_spotify_url, SONG_DATABASE


class TestRecommendSongs:
    def test_returns_list(self):
        result = recommend_songs("happy")
        assert isinstance(result, list)

    def test_correct_count(self):
        result = recommend_songs("happy", top_n=3)
        assert len(result) == 3

    def test_song_has_required_fields(self):
        songs = recommend_songs("calm", top_n=1)
        song = songs[0]
        for field in ["title", "artist", "genre", "mood_tags", "spotify_url"]:
            assert field in song, f"Missing field: {field}"

    def test_all_emotions_have_songs(self):
        for emotion in ["happy", "sad", "angry", "fearful", "surprised", "calm"]:
            songs = recommend_songs(emotion, top_n=1)
            assert len(songs) >= 1

    def test_invalid_emotion_raises(self):
        with pytest.raises(ValueError):
            recommend_songs("confused_beyond_measure")

    def test_spotify_url_format(self):
        songs = recommend_songs("happy", top_n=1)
        assert songs[0]["spotify_url"].startswith("https://open.spotify.com/search/")

    def test_mood_tags_is_list(self):
        songs = recommend_songs("sad", top_n=2)
        for song in songs:
            assert isinstance(song["mood_tags"], list)


class TestSpotifyUrl:
    def test_url_contains_title(self):
        url = get_spotify_url("Happy", "Pharrell")
        assert "Happy" in url or "happy" in url.lower()

    def test_url_starts_with_spotify(self):
        url = get_spotify_url("Test Song", "Test Artist")
        assert url.startswith("https://open.spotify.com")


class TestSongDatabase:
    def test_all_six_emotions_present(self):
        expected = {"happy", "sad", "angry", "fearful", "surprised", "calm"}
        assert set(SONG_DATABASE.keys()) == expected

    def test_each_emotion_has_at_least_five_songs(self):
        for emotion, songs in SONG_DATABASE.items():
            assert len(songs) >= 5, f"{emotion} has fewer than 5 songs"
