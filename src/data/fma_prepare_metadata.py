# src/data/fma_prepare_metadata.py
import os
import ast
import pandas as pd

BASE_DIR = "/work/NLP-mini-project/datasets/fma"
META_PATH = os.path.join(BASE_DIR, "fma_metadata", "fma_metadata")
TRACKS_PATH = os.path.join(META_PATH, "tracks.csv")
GENRES_PATH = os.path.join(META_PATH, "genres.csv")
AUDIO_BASE = os.path.join(BASE_DIR, "fma_large")
OUTPUT_DIR = "/work/NLP-mini-project/datasets/fma_clean"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clean_metadata.csv")

# IDs of the 12 top-level FMA genres used for classification
FINAL_TOP_LEVEL_IDS = [12, 15, 10, 17, 1235, 21, 2, 5, 4, 9, 3, 14]


def parse_list(s):
    return ast.literal_eval(s) if isinstance(s, str) else []


def build_audio_path(track_id):
    folder = f"{track_id // 1000:03d}"
    return os.path.join(AUDIO_BASE, folder, f"{track_id:06d}.mp3")


if __name__ == "__main__":
    tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
    genres = pd.read_csv(GENRES_PATH, index_col=0)

    tracks.columns = ["_".join(col).strip() for col in tracks.columns]
    df = tracks.reset_index().rename(columns={"index": "track_id"})

    df["genres_all"] = df["track_genres_all"].apply(parse_list)
    df = df[df["genres_all"].apply(len) > 0]

    id_to_title = genres["title"].to_dict()

    # Drop "Experimental" (genre 38) — too broad to be useful as a label
    df["genres_all"] = df["genres_all"].apply(lambda lst: [g for g in lst if g != 38])
    df["top_genres"] = df["genres_all"].apply(lambda lst: [g for g in lst if g in FINAL_TOP_LEVEL_IDS])
    df = df[df["top_genres"].apply(len) > 0]

    for g in FINAL_TOP_LEVEL_IDS:
        col = f"label_{id_to_title[g].replace(' ', '').replace('-', '')}"
        df[col] = df["top_genres"].apply(lambda lst: int(g in lst))

    label_cols = [c for c in df.columns if c.startswith("label_")]
    df["num_labels"] = df["top_genres"].apply(len)
    df["audio_path"] = df["track_id"].apply(build_audio_path)

    clean_df = df[["track_id", "audio_path", "top_genres", "num_labels"] + label_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(clean_df)} tracks to {OUTPUT_PATH}")
