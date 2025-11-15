import pandas as pd
import ast
import os

# ------------------------------------------------------------------------------------
# PATHS (UCloud version)
# ------------------------------------------------------------------------------------
BASE_DIR = "/work/NLP-mini-project/datasets/fma"

META_PATH = os.path.join(BASE_DIR, "fma_metadata", "fma_metadata")
TRACKS_PATH = os.path.join(META_PATH, "tracks.csv")
GENRES_PATH = os.path.join(META_PATH, "genres.csv")

# Path to the audio files (fma_large folder)
AUDIO_BASE = os.path.join(BASE_DIR, "fma_large")

# ------------------------------------------------------------------------------------
# Load metadata
# ------------------------------------------------------------------------------------
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
genres = pd.read_csv(GENRES_PATH, index_col=0)

# Flatten columns
tracks.columns = ["_".join(col).strip() for col in tracks.columns]
df = tracks.reset_index().rename(columns={"index": "track_id"})

# ------------------------------------------------------------------------------------
# Parse genres
# ------------------------------------------------------------------------------------
def parse_list(s):
    if isinstance(s, str):
        return ast.literal_eval(s)
    return []

df["genres_all"] = df["track_genres_all"].apply(parse_list)
df = df[df["genres_all"].apply(lambda lst: len(lst) > 0)]

# Final top genres
final_top_level_ids = [
    12, 15, 10, 17, 1235,
    21, 2, 5, 4, 9, 3, 14
]

id_to_title = genres["title"].to_dict()

# Remove experimental (38)
df["genres_all"] = df["genres_all"].apply(lambda lst: [g for g in lst if g != 38])

# Extract top-level genres
df["top_genres"] = df["genres_all"].apply(
    lambda lst: [g for g in lst if g in final_top_level_ids]
)
df = df[df["top_genres"].apply(len) > 0]

# ------------------------------------------------------------------------------------
# Create binary label columns
# ------------------------------------------------------------------------------------
for g in final_top_level_ids:
    title = id_to_title[g]
    colname = f"label_{title.replace(' ', '').replace('-', '')}"
    df[colname] = df["top_genres"].apply(lambda lst: int(g in lst))

label_cols = [c for c in df.columns if c.startswith("label_")]
df["num_labels"] = df["top_genres"].apply(len)

# ------------------------------------------------------------------------------------
# Build audio paths
# ------------------------------------------------------------------------------------
def build_audio_path(track_id):
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return os.path.join(AUDIO_BASE, folder, filename)

df["audio_path"] = df["track_id"].apply(build_audio_path)

# ------------------------------------------------------------------------------------
# Select final columns
# ------------------------------------------------------------------------------------
clean_df = df[["track_id", "audio_path", "top_genres", "num_labels"] + label_cols]

# ------------------------------------------------------------------------------------
# Save output
# ------------------------------------------------------------------------------------
OUTPUT_DIR = "/work/NLP-mini-project/datasets/fma_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clean_metadata.csv")
clean_df.to_csv(OUTPUT_PATH, index=False)

print("Clean metadata saved to:", OUTPUT_PATH)
print("Number of tracks:", len(clean_df))
