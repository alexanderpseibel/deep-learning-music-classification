import pandas as pd
import ast
import os

# Paths
META_PATH = "/Member Files: AlexanderPhilippSeibel#0732 (1098394)/NLP-mini-project/datasets/fma/fma_metadata/fma_metadata/"
TRACKS_PATH = os.path.join(META_PATH, "tracks.csv")
GENRES_PATH = os.path.join(META_PATH, "genres.csv")

AUDIO_BASE = "fma_large"

# Load metadata
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
genres = pd.read_csv(GENRES_PATH, index_col=0)

# Flatten tracks.csv columns
tracks.columns = ["_".join(col).strip() for col in tracks.columns]
df = tracks.reset_index().rename(columns={"index": "track_id"})

# Parse genre lists from string to list
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

# Remove Experimental (38)
df["genres_all"] = df["genres_all"].apply(lambda lst: [g for g in lst if g != 38])

# Extract top-level genres
df["top_genres"] = df["genres_all"].apply(
    lambda lst: [g for g in lst if g in final_top_level_ids]
)
df = df[df["top_genres"].apply(len) > 0]

# Create binary label columns
for g in final_top_level_ids:
    title = id_to_title[g]
    colname = f"label_{title.replace(' ', '').replace('-', '')}"
    df[colname] = df["top_genres"].apply(lambda lst: int(g in lst))

label_cols = [c for c in df.columns if c.startswith("label_")]

df["num_labels"] = df["top_genres"].apply(len)

# Build audio paths
def build_audio_path(track_id):
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    return f"{AUDIO_BASE}/{folder}/{filename}"

df["audio_path"] = df["track_id"].apply(build_audio_path)

# Select final columns
clean_df = df[["track_id", "audio_path", "top_genres", "num_labels"] + label_cols]

# Save
OUTPUT_DIR = "/work/fma_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

clean_df.to_csv(f"{OUTPUT_DIR}/clean_metadata.csv", index=False)

print("Clean metadata saved to:", f"{OUTPUT_DIR}/clean_metadata.csv")
print("Number of tracks:", len(clean_df))
