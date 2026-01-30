import sys
import time
import torch

from .model import TinyAudioCNN
from .config import MODELS_DIR, CLASSES
from .mfcc_extractor import load_audio_mono, audio_to_mfcc

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.infer path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]

    ckpt_path = MODELS_DIR / "audio_event_cnn.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = TinyAudioCNN(num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.eval()

    t0 = time.time()
    y, _ = load_audio_mono(wav_path)
    mfcc = audio_to_mfcc(y)
    x = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)  # (1,1,40,T)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
    dt = (time.time() - t0) * 1000

    print({"prediction": CLASSES[pred], "latency_ms": round(dt, 2)})

if __name__ == "__main__":
    main()
