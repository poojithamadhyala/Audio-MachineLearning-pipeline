import os
import tarfile
import urllib.request
from pathlib import Path

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[OK] Already downloaded: {out_path}")
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK] Saved: {out_path}")

def extract(tar_path: Path, extract_to: Path):
    marker = extract_to / ".extracted"
    if marker.exists():
        print(f"[OK] Already extracted: {extract_to}")
        return
    print(f"Extracting {tar_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    marker.write_text("done")
    print("[OK] Extracted.")

def main():
    raw_dir = Path("data/raw")
    tgz = raw_dir / "speech_commands_v0.02.tar.gz"
    extract_dir = raw_dir / "speech_commands_v0.02"

    download(URL, tgz)
    extract(tgz, extract_dir)

    print("\nNext:")
    print("  python src/train.py")

if __name__ == "__main__":
    main()
