# src/data/fma_transform_to_spectrograms.py
import os
import traceback
import numpy as np
import librosa
from multiprocessing import Pool
from tqdm import tqdm

INPUT_ROOT = "/datasets/fma/fma_clean/fma_selected"
OUTPUT_ROOT = "/work/NLP-mini-project/mels"

SAMPLE_RATE = 32000
N_MELS = 128
WIN_LENGTH = int(0.025 * SAMPLE_RATE)  # 25ms
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 10ms


def process_file(path_tuple):
    in_path, out_path = path_tuple

    if os.path.exists(out_path):
        return f"SKIPPED {out_path}"

    try:
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0,
        )
        np.save(out_path, librosa.power_to_db(mel, ref=np.max))
        return f"OK {out_path}"
    except Exception as e:
        msg = f"ERROR {in_path}: {repr(e)}"
        with open(os.path.join(OUTPUT_ROOT, "errors.log"), "a") as f:
            f.write(msg + "\n" + traceback.format_exc() + "\n")
        return msg


if __name__ == "__main__":
    # FMA uses 000–255 subfolders
    for i in range(256):
        os.makedirs(os.path.join(OUTPUT_ROOT, f"{i:03d}"), exist_ok=True)

    tasks = []
    for root, dirs, files in os.walk(INPUT_ROOT):
        for f in files:
            if not f.endswith(".mp3"):
                continue
            track_id = f[:-4]       # e.g. "097023"
            subfolder = track_id[:3]
            tasks.append((
                os.path.join(root, f),
                os.path.join(OUTPUT_ROOT, subfolder, track_id + ".npy"),
            ))

    print(f"Found {len(tasks)} audio files.")

    num_workers = os.cpu_count() - 1 or 1
    print(f"Using {num_workers} workers...")
    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), ncols=100):
            pass

    print(f"Done. Mels saved to: {OUTPUT_ROOT}")
