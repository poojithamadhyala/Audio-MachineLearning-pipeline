import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from .dataset import get_datasets
from .model import TinyAudioCNN
from .config import MODELS_DIR, CLASSES

def plot_cm(cm, labels, out_path="models/confusion_matrix.png"):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[OK] Saved confusion matrix to {out_path}")

def main():
    train_ds, val_ds, test_ds, _ = get_datasets(max_per_class=3000)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    ckpt = torch.load(MODELS_DIR / "audio_event_cnn.pt", map_location="cpu")

    model = TinyAudioCNN(num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())

    # ✅ Force sklearn to include all classes even if some are missing in this split
    labels = list(range(len(CLASSES)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    ))

    plot_cm(cm, CLASSES)


if __name__ == "__main__":
    main()
