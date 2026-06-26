"""
music_recommender.py
---------------------
Given a detected emotion, returns a curated list of song recommendations.

Each song entry contains:
    title, artist, genre, mood_tags, spotify_search_url
"""

from typing import List, Dict

# ---------------------------------------------------------------------------
# Song database  (easily extendable — just add more dicts!)
# ---------------------------------------------------------------------------

SONG_DATABASE: Dict[str, List[Dict]] = {
    "happy": [
        {"title": "Happy",                "artist": "Pharrell Williams",   "genre": "Pop",       "mood_tags": ["upbeat", "cheerful", "positive"]},
        {"title": "Can't Stop the Feeling","artist": "Justin Timberlake",  "genre": "Pop/Dance", "mood_tags": ["joyful", "dance", "energetic"]},
        {"title": "Good as Hell",          "artist": "Lizzo",              "genre": "Pop/R&B",   "mood_tags": ["empowering", "uplifting", "fun"]},
        {"title": "Shake It Off",          "artist": "Taylor Swift",       "genre": "Pop",       "mood_tags": ["carefree", "bubbly", "energetic"]},
        {"title": "Uptown Funk",           "artist": "Mark Ronson ft. Bruno Mars", "genre": "Funk/Pop", "mood_tags": ["groovy", "confident", "fun"]},
        {"title": "Walking on Sunshine",   "artist": "Katrina & The Waves","genre": "Pop/Rock",  "mood_tags": ["radiant", "joyful", "classic"]},
        {"title": "I Gotta Feeling",       "artist": "The Black Eyed Peas","genre": "Pop/Dance", "mood_tags": ["party", "hopeful", "celebratory"]},
    ],
    "sad": [
        {"title": "Someone Like You",      "artist": "Adele",              "genre": "Pop/Soul",  "mood_tags": ["heartbreak", "longing", "emotional"]},
        {"title": "The Night We Met",      "artist": "Lord Huron",         "genre": "Indie Folk","mood_tags": ["nostalgic", "melancholy", "bittersweet"]},
        {"title": "Skinny Love",           "artist": "Bon Iver",           "genre": "Indie Folk","mood_tags": ["raw", "quiet grief", "introspective"]},
        {"title": "Fix You",               "artist": "Coldplay",           "genre": "Alternative","mood_tags": ["comforting", "hopeful sorrow", "tender"]},
        {"title": "Breathe Me",            "artist": "Sia",                "genre": "Art Pop",   "mood_tags": ["fragile", "desperate", "emotional"]},
        {"title": "Liability",             "artist": "Lorde",              "genre": "Indie Pop", "mood_tags": ["isolated", "reflective", "honest"]},
        {"title": "Hurt",                  "artist": "Johnny Cash",        "genre": "Country",   "mood_tags": ["regret", "pain", "poignant"]},
    ],
    "angry": [
        {"title": "Break Stuff",           "artist": "Limp Bizkit",        "genre": "Nu-Metal",  "mood_tags": ["aggression", "release", "loud"]},
        {"title": "Killing in the Name",   "artist": "Rage Against the Machine", "genre": "Rock/Metal", "mood_tags": ["rebel", "powerful", "furious"]},
        {"title": "Given Up",              "artist": "Linkin Park",        "genre": "Rock",      "mood_tags": ["frustration", "intense", "cathartic"]},
        {"title": "You Oughta Know",       "artist": "Alanis Morissette",  "genre": "Alt Rock",  "mood_tags": ["rage", "betrayal", "fierce"]},
        {"title": "Blow",                  "artist": "Beyonce",            "genre": "R&B/Pop",   "mood_tags": ["fierce", "powerful", "bold"]},
        {"title": "Stronger",              "artist": "Kanye West",         "genre": "Hip-Hop",   "mood_tags": ["defiant", "driven", "hard"]},
        {"title": "Since U Been Gone",     "artist": "Kelly Clarkson",     "genre": "Pop/Rock",  "mood_tags": ["venting", "liberation", "empowered"]},
    ],
    "fearful": [
        {"title": "Fear",                  "artist": "Kendrick Lamar",     "genre": "Hip-Hop",   "mood_tags": ["anxiety", "introspective", "raw"]},
        {"title": "Heavy",                 "artist": "Birdtalker",         "genre": "Indie Folk","mood_tags": ["letting go", "gentle", "soothing fear"]},
        {"title": "Safe and Sound",        "artist": "Taylor Swift ft. Civil Wars", "genre": "Folk Pop", "mood_tags": ["comfort", "reassurance", "soft"]},
        {"title": "Breathe (2 AM)",        "artist": "Anna Nalick",        "genre": "Pop/Folk",  "mood_tags": ["overwhelm", "coping", "tender"]},
        {"title": "The Sound of Silence",  "artist": "Disturbed",          "genre": "Rock",      "mood_tags": ["dread", "powerful", "cinematic"]},
        {"title": "Lean on Me",            "artist": "Bill Withers",       "genre": "Soul",      "mood_tags": ["comforting", "supportive", "warm"]},
        {"title": "Shake It Out",          "artist": "Florence + The Machine", "genre": "Indie Rock", "mood_tags": ["facing fear", "anthemic", "hopeful"]},
    ],
    "surprised": [
        {"title": "What's Going On",       "artist": "Marvin Gaye",        "genre": "Soul",      "mood_tags": ["reflective", "wondering", "classic"]},
        {"title": "Somebody That I Used to Know", "artist": "Gotye",       "genre": "Indie Pop", "mood_tags": ["disbelief", "unexpected", "emotional"]},
        {"title": "Mr. Brightside",        "artist": "The Killers",        "genre": "Indie Rock","mood_tags": ["sudden rush", "vivid", "energetic"]},
        {"title": "Counting Stars",        "artist": "OneRepublic",        "genre": "Pop/Rock",  "mood_tags": ["amazed", "hopeful", "dynamic"]},
        {"title": "As It Was",             "artist": "Harry Styles",       "genre": "Pop",       "mood_tags": ["unexpected change", "bittersweet", "catchy"]},
        {"title": "Upside Down",           "artist": "Jack Johnson",       "genre": "Acoustic",  "mood_tags": ["wonder", "discovery", "light"]},
        {"title": "Beautiful Day",         "artist": "U2",                 "genre": "Rock",      "mood_tags": ["awe", "uplifting", "expansive"]},
    ],
    "calm": [
        {"title": "Holocene",              "artist": "Bon Iver",           "genre": "Indie Folk","mood_tags": ["peaceful", "vast", "meditative"]},
        {"title": "Clair de Lune",         "artist": "Claude Debussy",     "genre": "Classical", "mood_tags": ["tranquil", "timeless", "dreamy"]},
        {"title": "Gymnopédie No.1",       "artist": "Erik Satie",         "genre": "Classical", "mood_tags": ["serene", "slow", "beautiful"]},
        {"title": "Breathe",               "artist": "Pink Floyd",         "genre": "Rock",      "mood_tags": ["spacious", "slow", "reflective"]},
        {"title": "The Night",             "artist": "Zac Brown Band",     "genre": "Country",   "mood_tags": ["easy", "warm", "comfortable"]},
        {"title": "Golden Hour",           "artist": "JVKE",               "genre": "Pop",       "mood_tags": ["warm", "soft", "romantic calm"]},
        {"title": "Weightless",            "artist": "Marconi Union",      "genre": "Ambient",   "mood_tags": ["stress relief", "floating", "therapeutic"]},
    ],
}

# Spotify search base URL (opens search in browser — no API key needed)
SPOTIFY_SEARCH_BASE = "https://open.spotify.com/search/"


def get_spotify_url(title: str, artist: str) -> str:
    """Builds a Spotify search URL for a song."""
    import urllib.parse
    query = urllib.parse.quote(f"{title} {artist}")
    return f"{SPOTIFY_SEARCH_BASE}{query}"


def recommend_songs(emotion: str, top_n: int = 5) -> List[Dict]:
    """
    Returns top_n song recommendations for the given emotion.
    Also injects Spotify search URLs into each entry.
    """
    emotion = emotion.lower().strip()

    if emotion not in SONG_DATABASE:
        available = ", ".join(SONG_DATABASE.keys())
        raise ValueError(f"Unknown emotion '{emotion}'. Available: {available}")

    songs = SONG_DATABASE[emotion][:top_n]

    # Enrich each song with a Spotify URL
    enriched = []
    for song in songs:
        entry = song.copy()
        entry["spotify_url"] = get_spotify_url(song["title"], song["artist"])
        enriched.append(entry)

    return enriched


def display_recommendations(emotion: str, songs: List[Dict]) -> None:
    """Pretty-prints the song recommendations to the console."""
    emoji_map = {
        "happy": "😊", "sad": "😢", "angry": "😠",
        "fearful": "😨", "surprised": "😲", "calm": "😌",
    }
    emoji = emoji_map.get(emotion, "🎵")

    print(f"\n{emoji}  You're feeling {emotion.upper()} — here are your songs:\n")
    print("-" * 60)
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song['title']}  —  {song['artist']}")
        print(f"   Genre : {song['genre']}")
        print(f"   Vibe  : {', '.join(song['mood_tags'])}")
        print(f"   Listen: {song['spotify_url']}")
        print()


if __name__ == "__main__":
    for emotion in ["happy", "sad", "calm"]:
        songs = recommend_songs(emotion, top_n=3)
        display_recommendations(emotion, songs)
