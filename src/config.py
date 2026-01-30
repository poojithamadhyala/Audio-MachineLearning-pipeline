from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "speech_commands_v0.02"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Keep it small + wearable-friendly
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)

N_MFCC = 40
HOP_LENGTH = 160   # 10ms at 16kHz
N_FFT = 400        # 25ms window

# Classes for this project
CLASSES = ["speech", "noise", "silence"]
