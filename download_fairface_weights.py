from __future__ import annotations
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

# Change this only if your repo/file name differs
REPO_ID = "face-attributes/fairface"
REPO_TYPE = "dataset"
FILENAME = "fairface_res34.pth"

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    token = os.getenv("HF_TOKEN", None)
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. In PowerShell run:\n"
            "  $env:HF_TOKEN=\"hf_...\"\n"
            "Then re-run this script."
        )

    print("⬇️ Downloading FairFace checkpoint from Hugging Face...")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=FILENAME,
        token=token,
        local_dir=str(OUT_DIR),
        local_dir_use_symlinks=False,  # Windows-friendly
    )

    # Ensure final expected name/path
    final_path = OUT_DIR / FILENAME
    if str(local_path) != str(final_path):
        # hf_hub_download might place it in subfolders; copy/rename to expected location
        final_path.write_bytes(Path(local_path).read_bytes())

    print(f"✅ Saved to: {final_path}")

if __name__ == "__main__":
    main()
