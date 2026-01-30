import torch
from .model import TinyAudioCNN
from .config import MODELS_DIR, CLASSES

def main():
    ckpt = torch.load(MODELS_DIR / "audio_event_cnn.pt", map_location="cpu")

    model = TinyAudioCNN(num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 1, 40, 101)  # (B, C, MFCC, time)
    out_path = MODELS_DIR / "audio_event_cnn.onnx"

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["mfcc"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"mfcc": {0: "batch"}}
    )

    print(f"[OK] Exported ONNX model to {out_path}")

if __name__ == "__main__":
    main()
