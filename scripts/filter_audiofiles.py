import os
import shutil
import pandas as pd

# Paths
CLEAN_META = "/work/NLP-mini-project/datasets/fma_clean/clean_metadata.csv"

SRC_AUDIO = "/work/NLP-mini-project/datasets/fma/fma_mp3s/fma_large"
DST_AUDIO = "/work/NLP-mini-project/datasets/fma/fma_selected"

# Load metadata
df = pd.read_csv(CLEAN_META)

# Create destination directory
os.makedirs(DST_AUDIO, exist_ok=True)

missing_files = 0
copied_files = 0

for _, row in df.iterrows():
    track_id = row["track_id"]

    # Build source path correctly
    folder = f"{track_id // 1000:03d}"
    filename = f"{track_id:06d}.mp3"
    src_path = os.path.join(SRC_AUDIO, folder, filename)

    # Build destination directory
    dst_dir = os.path.join(DST_AUDIO, folder)
    os.makedirs(dst_dir, exist_ok=True)

    dst_path = os.path.join(dst_dir, filename)

    # Copy file
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_files += 1
    else:
        missing_files += 1

print(f"Copied {copied_files} audio files.")
print(f"Missing {missing_files} audio files (likely corrupted or missing).")
print("Filtered dataset saved to:", DST_AUDIO)
