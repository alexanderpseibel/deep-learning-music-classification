#src/data/fma_transform_to_spectrograms.py
import os
import librosa
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import traceback

# -------------------------------------------------------
# CONFIG — UPDATE THESE TWO PATHS
# -------------------------------------------------------
INPUT_ROOT = "/datasets/fma/fma_clean/fma_selected"
OUTPUT_ROOT = "/work/NLP-mini-project/mels"      

SAMPLE_RATE = 32000
N_MELS = 128
WIN_LENGTH = int(0.025 * SAMPLE_RATE)  # 25ms window
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 10ms hop


# -------------------------------------------------------
# CREATE OUTPUT FOLDER STRUCTURE (000, 001, ...)
# -------------------------------------------------------
for i in range(256):  # FMA small/medium/large use 000–255
    folder = os.path.join(OUTPUT_ROOT, f"{i:03d}")
    os.makedirs(folder, exist_ok=True)


def process_file(path_tuple):
    """ Worker function for multiprocessing. """
    in_path, out_path = path_tuple

    # Skip if already processed
    if os.path.exists(out_path):
        return f"SKIPPED {out_path}"

    try:
        # Load audio
        y, sr = librosa.load(in_path, sr=SAMPLE_RATE, mono=True)

        # Compute mel
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Save output
        np.save(out_path, mel_db)

        return f"OK {out_path}"

    except Exception as e:
        error_msg = f"ERROR {in_path}: {repr(e)}"
        with open(os.path.join(OUTPUT_ROOT, "errors.log"), "a") as f:
            f.write(error_msg + "\n")
            f.write(traceback.format_exc() + "\n")
        return error_msg


# -------------------------------------------------------
# GATHER ALL MP3 FILES + MAP TO OUTPUT PATH
# -------------------------------------------------------
tasks = []

for root, dirs, files in os.walk(INPUT_ROOT):
    for f in files:
        if not f.endswith(".mp3"):
            continue

        # Example: /datasets/.../fma_selected/097/097023.mp3
        track_id = f[:-4]              # "097023"
        subfolder = track_id[:3]       # "097"

        in_path = os.path.join(root, f)
        out_path = os.path.join(OUTPUT_ROOT, subfolder, track_id + ".npy")

        tasks.append((in_path, out_path))

print(f"Found {len(tasks)} audio files to process.")


# -------------------------------------------------------
# MULTIPROCESSING EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":
    num_workers = os.cpu_count() - 1 or 1
    print(f"Using {num_workers} workers...")

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_file, tasks),
                           total=len(tasks), ncols=100):
            pass

    print("Done! All mels saved to:", OUTPUT_ROOT)
