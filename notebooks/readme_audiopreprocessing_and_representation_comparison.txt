Audio Preprocessing & Representation Comparison
--------------------------------------------------------
Notebook 1 — Audio Preprocessing

Purpose:
This notebook converts each music track into three standardized audio representations:

Waveform (raw audio, 44.1 kHz, 30 seconds → 1,323,000 samples)

Mel Spectrogram (128 Mel bins, ~3000 time frames)

CQT Spectrogram (128 frequency bins, ~3000 time frames)


Key steps:

Load MP3 files from the Free Music Archive dataset

Resample, normalize, trim/pad to a fixed 30-second duration

Compute Mel and CQT spectrograms

Pad or trim all spectrograms to a consistent shape: (128, 3000)

Save all arrays as .npy files

Create a final dataframe df_audio.xlsx linking each track to its waveform, Mel, and CQT paths

The resulting features are used in the next notebook for model training.

--------------------------------------------------------------
Notebook 2 — Representation Comparison with CNN Models

Purpose:
Building on the preprocessed features, this notebook compares four audio representations:

Waveform (1D input)

Mel Spectrogram

CQT Spectrogram

Multichannel Spectrogram (Mel + CQT stacked as 2 channels)


What the notebook does:

Stratified multi-label train/val/test split

Define two model architectures:

1D CNN for waveform

2D CNN for Mel, CQT, and multichannel inputs

Train each representation for 10 epochs with early stopping

Evaluate using F1-micro and F1-macro scores

Save and reload models + training histories

Compare model performance across representations


Results:
The models were evaluated using F1-Micro and F1-Macro scores, where F1-Micro measures performance weighted by label frequency (overall correctness), and F1-Macro averages the F1 over all genres equally, treating frequent and rare genres the same.

Exact results:

Waveform (1D-CNN):

F1 Micro: 0.0549

F1 Macro: 0.0400

Mel Spectrogram:

F1 Micro: 0.1186

F1 Macro: 0.0975

CQT Spectrogram:

F1 Micro: 0.0883

F1 Macro: 0.0721

Multichannel (Mel + CQT):

F1 Micro: 0.1050

F1 Macro: 0.0914

Conclusion:
The Mel spectrogram achieves the highest performance among all representations, which aligns with findings in audio research where Mel-based features are widely known to be effective for music genre classification and general audio tagging.