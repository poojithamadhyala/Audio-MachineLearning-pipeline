import time
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .dataset import get_datasets
from .model import TinyAudioCNN
from .config import MODELS_DIR, CLASSES

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def eval_loop(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
    return accuracy_score(y_true, y_pred)

def main():
    device = get_device()
    print(f"Using device: {device}")

    print("Loading dataset (Speech Commands v0.02)...")
    train_ds, val_ds, test_ds, label_to_idx = get_datasets(max_per_class=3000)  # adjust if needed

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TinyAudioCNN(num_classes=len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = MODELS_DIR / "audio_event_cnn.pt"

    for epoch in range(1, 6):
        model.train()
        t0 = time.time()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_acc = eval_loop(model, val_loader, device)
        dt = time.time() - t0
        print(f"Epoch {epoch} | loss={sum(losses)/len(losses):.4f} | val_acc={val_acc:.4f} | time={dt:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "label_to_idx": label_to_idx}, best_path)
            print(f"[OK] Saved best model to {best_path}")

    # final test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc = eval_loop(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print("Labels:", CLASSES)

if __name__ == "__main__":
    main()
