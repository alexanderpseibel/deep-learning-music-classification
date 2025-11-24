#src/data/fma_filter_audiofiles.py
#!/usr/bin/env python3
import os
import shutil
import pandas as pd
from multiprocessing import Pool, cpu_count

CLEAN_META = "/work/NLP-mini-project/datasets/fma_clean/clean_metadata.csv"
SRC_AUDIO = "/work/NLP-mini-project/datasets/fma/fma_mp3s/fma_large"
DST_AUDIO = "/work/NLP-mini-project/datasets/fma/fma_selected"

# Load metadata
df = pd.read_csv(CLEAN_META)

# Ensure destination base exists
os.makedirs(DST_AUDIO, exist_ok=True)

def copy_file(track_id):
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"

    src = os.path.join(SRC_AUDIO, folder, filename)
    dst_dir = os.path.join(DST_AUDIO, folder)
    dst = os.path.join(dst_dir, filename)

    os.makedirs(dst_dir, exist_ok=True)

    if os.path.exists(src):
        shutil.copy2(src, dst)
        return 1
    else:
        return 0

if __name__ == "__main__":
    track_ids = df["track_id"].tolist()

    workers = min(32, cpu_count())  # use max 32 workers
    print(f"Using {workers} parallel workers...")

    copied = 0

    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(copy_file, track_ids), start=1):
            copied += result
            if i % 1000 == 0:
                print(f"Copied {copied}/{i} files so far...")

    print(f"\nFinished!")
    print(f"Total copied: {copied}")
    print(f"Missing files: {len(track_ids) - copied}")
