import time
import numpy as np
import onnxruntime as ort

from .config import MODELS_DIR

def main():
    model_path = (MODELS_DIR / "audio_event_cnn.onnx").as_posix()
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    x = np.random.randn(1, 1, 40, 101).astype(np.float32)

    # warmup
    for _ in range(10):
        sess.run(None, {"mfcc": x})

    runs = 100
    t0 = time.time()
    for _ in range(runs):
        sess.run(None, {"mfcc": x})
    dt = (time.time() - t0) * 1000 / runs

    print(f"Average ONNX inference latency: {dt:.3f} ms (batch=1)")

if __name__ == "__main__":
    main()
