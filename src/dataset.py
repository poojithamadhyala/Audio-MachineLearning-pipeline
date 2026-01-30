import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DATA_RAW, CLASSES
from .mfcc_extractor import load_audio_mono, audio_to_mfcc

# SpeechCommands has many folders; we use:
# - "yes", "no", "up", ... (words) as "speech"
# - "_background_noise_" as "noise"
# - silence: synthetic zeros + sometimes low-energy slices from background noise

SPEECH_FOLDERS = [
    "yes","no","up","down","left","right","on","off","stop","go",
    "one","two","three","four","five","six","seven","eight","nine","zero"
]

class AudioEventDataset(Dataset):
    def __init__(self, items, label_to_idx):
        self.items = items
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        if label == "silence":
            # synthetic silence
            y = np.zeros(16000, dtype=np.float32)
        else:
            y, _ = load_audio_mono(str(path))

        mfcc = audio_to_mfcc(y)  # (40, T)
        x = torch.from_numpy(mfcc).unsqueeze(0)  # (1, 40, T)
        y_idx = torch.tensor(self.label_to_idx[label], dtype=torch.long)
        return x, y_idx

def list_wavs(folder: Path):
    return list(folder.rglob("*.wav"))

def build_items(seed=42, max_per_class=None):
    random.seed(seed)

    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_RAW}\n"
            f"Run: python download_data.py"
        )

    items = []

    # 1) speech
    speech_paths = []
    for name in SPEECH_FOLDERS:
        d = DATA_RAW / name
        if d.exists():
            speech_paths.extend(list_wavs(d))
    random.shuffle(speech_paths)
    if max_per_class:
        speech_paths = speech_paths[:max_per_class]
    items.extend([(p, "speech") for p in speech_paths])

    # 2) noise (background noise wavs)
    noise_dir = DATA_RAW / "_background_noise_"
    noise_paths = list_wavs(noise_dir) if noise_dir.exists() else []
    random.shuffle(noise_paths)
    if max_per_class and noise_paths:
        noise_paths = noise_paths[:max(1, max_per_class // 5)]
    items.extend([(p, "noise") for p in noise_paths])

    # 3) silence (synthetic entries)
    # Make silence count similar scale to noise for balance
    silence_count = max(200, len(noise_paths) * 10) if noise_paths else 500
    if max_per_class:
        silence_count = min(silence_count, max_per_class)
    items.extend([(None, "silence") for _ in range(silence_count)])

    random.shuffle(items)
    return items

def split_items(items, train=0.8, val=0.1):
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    train_items = items[:n_train]
    val_items = items[n_train:n_train+n_val]
    test_items = items[n_train+n_val:]
    return train_items, val_items, test_items

def get_datasets(max_per_class=None):
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    items = build_items(max_per_class=max_per_class)

    train_items, val_items, test_items = split_items(items)

    train_ds = AudioEventDataset(train_items, label_to_idx)
    val_ds = AudioEventDataset(val_items, label_to_idx)
    test_ds = AudioEventDataset(test_items, label_to_idx)

    return train_ds, val_ds, test_ds, label_to_idx
