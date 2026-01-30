import numpy as np
import librosa
from .config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, NUM_SAMPLES

def load_audio_mono(path: str):
    # librosa load handles wav decoding without ffmpeg headaches for SpeechCommands
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    # pad/trim to fixed length
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:NUM_SAMPLES]
    return y, sr

def audio_to_mfcc(y: np.ndarray):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    # Normalize per-sample for stability
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    # Shape: (n_mfcc, time)
    return mfcc.astype(np.float32)
