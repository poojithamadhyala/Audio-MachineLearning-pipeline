import time
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import torch

from src.model import TinyAudioCNN
from src.config import MODELS_DIR, CLASSES
from src.mfcc_extractor import load_audio_mono, audio_to_mfcc

app = FastAPI(title="Audio Event Classifier")

ckpt_path = MODELS_DIR / "audio_event_cnn.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

model = TinyAudioCNN(num_classes=len(CLASSES))
model.load_state_dict(ckpt["model"])
model.eval()

TMP_DIR = Path("data/processed")
TMP_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"status": "ok", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()

    tmp_path = TMP_DIR / "upload.wav"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    y, _ = load_audio_mono(str(tmp_path))
    mfcc = audio_to_mfcc(y)
    x = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    dt = (time.time() - t0) * 1000
    return {"prediction": CLASSES[pred], "latency_ms": round(dt, 2)}
