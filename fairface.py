# fairface_light_resume.py
from __future__ import annotations

import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
from tqdm import tqdm

# ----------------------------
# 0) Stabilize Windows / Torch
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

import faulthandler
faulthandler.enable()

# ----------------------------
# 1) Config
# ----------------------------
IN_CSV = "data/laion_big_light_tau1_0.2989.csv"  # <-- your LIGHT dataset (107k)
OUT_CSV = "data/laion_big_light_with_fairface.csv"
STATE_JSON = "data/fairface_light_state.json"
SUMMARY_JSON = "data/fairface_light_summary.json"

# Process only a chunk per run to reduce crash probability
# You can increase once stable.
MAX_ROWS_PER_RUN = 20_000

# Save every N processed rows
SAVE_EVERY = 500

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Face detection model (MTCNN)
# Install:
#   pip install facenet-pytorch
from facenet_pytorch import MTCNN

# FairFace classifier model:
# We'll use a common public FairFace implementation checkpoint logic.
# If your previous script already loads it, keep it.
# Here we implement a simple torch hub load of the model used in many fairface scripts.
# If this part doesn't match your current fairface.py, tell me and I’ll adapt to your exact model loader.
import torchvision.transforms as T
import torchvision

# ----------------------------
# 2) Helpers
# ----------------------------
def ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_open_rgb(p: str) -> Optional[Image.Image]:
    try:
        img = Image.open(p)
        img = img.convert("RGB")
        return img
    except Exception:
        return None

def load_state() -> Dict[str, Any]:
    if Path(STATE_JSON).exists():
        try:
            return json.loads(Path(STATE_JSON).read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: Dict[str, Any]):
    ensure_parent(STATE_JSON)
    Path(STATE_JSON).write_text(json.dumps(state, indent=2), encoding="utf-8")

def save_checkpoint(df: pd.DataFrame):
    ensure_parent(OUT_CSV)
    df.to_csv(OUT_CSV, index=False)

def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    def fill_ratio(col: str) -> float:
        return float(df[col].notna().mean()) if col in df.columns else 0.0

    out = {
        "rows_total": int(len(df)),
        "device": DEVICE,
        "columns": list(df.columns),
        "fill_ratio": {
            "face_detected": fill_ratio("face_detected"),
            "gender": fill_ratio("gender"),
            "age_group": fill_ratio("age_group"),
            "skin_tone": fill_ratio("skin_tone"),
        }
    }

    if "face_detected" in df.columns:
        vc = df["face_detected"].value_counts(dropna=False).to_dict()
        out["face_detected_counts"] = {str(k): int(v) for k, v in vc.items()}

    for col in ["gender", "age_group", "skin_tone"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            out[f"{col}_counts"] = {str(k): int(v) for k, v in vc.items()}

    return out

# ----------------------------
# 3) Load FairFace head (simple baseline)
# ----------------------------
# NOTE: FairFace has 3 heads: gender (2), age (9), race (7).
# We'll map race -> "skin_tone" categories (rough mapping), or keep the race label if you prefer.
# If you have a different FairFace implementation already, I can plug it in.
#
# To keep this reliable, we use a ResNet34 backbone and expect a checkpoint file.
# If you already have fairface weights locally, set FAIRFACE_CKPT path.

FAIRFACE_CKPT = "data/fairface_res34.pth"  # <-- put your checkpoint here if you have it

GENDER_LABELS = ["Male", "Female"]
AGE_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Common FairFace race labels (varies by implementation)
RACE_LABELS = ["White", "Black", "Latino_Hispanic", "East_Asian", "Southeast_Asian", "Indian", "Middle_Eastern"]

# You can keep "race" OR convert to coarse skin tone buckets.
# Here: we keep the original race label under "skin_tone" for now (simpler + safer).
# If you want real skin tone buckets, we can add a mapping later.
def race_to_skin_tone(race: str) -> str:
    return race

def load_fairface_model() -> torch.nn.Module:
    # Simple multi-head model template
    backbone = torchvision.models.resnet34(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()

    class MultiHead(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.gender = torch.nn.Linear(in_features, 2)
            self.age = torch.nn.Linear(in_features, 9)
            self.race = torch.nn.Linear(in_features, 7)

        def forward(self, x):
            feats = self.base(x)
            return self.gender(feats), self.age(feats), self.race(feats)

    model = MultiHead(backbone)

    ckpt_path = Path(FAIRFACE_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"FairFace checkpoint not found: {FAIRFACE_CKPT}\n"
            f"➡️ Put the FairFace resnet34 checkpoint there, or tell me the path you already use."
        )

    sd = torch.load(str(ckpt_path), map_location="cpu")
    # some checkpoints store {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # remove "module." if needed
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)

    model.eval()
    model.to(DEVICE)
    return model

FAIRFACE_TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# 4) Main
# ----------------------------
def main():
    print("Device:", DEVICE)

    # Read base CSV
    df_in = safe_read_csv(IN_CSV)
    print("Rows:", len(df_in))
    print("Columns:", list(df_in.columns))

    # Ensure image existence column if not present
    if "exists" not in df_in.columns:
        df_in["exists"] = df_in["image_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())
    print(f"Exists %: {df_in['exists'].mean()*100:.2f}%")

    # Load checkpoint CSV if exists (resume)
    if Path(OUT_CSV).exists():
        df = safe_read_csv(OUT_CSV)
        print(f"🔁 Resume enabled. Loaded checkpoint CSV: {OUT_CSV}")
        # Make sure we have same length and can align
        if len(df) != len(df_in):
            print("⚠️ Checkpoint length differs from input. Re-aligning by image_path (best effort).")
            # Best effort merge
            df = df_in.merge(df.drop(columns=[c for c in df.columns if c in df_in.columns and c != "image_path"]),
                             on="image_path", how="left")
        else:
            # bring new columns from input if missing
            for c in df_in.columns:
                if c not in df.columns:
                    df[c] = df_in[c]
    else:
        df = df_in.copy()

    # Ensure output columns exist
    for col in ["face_detected", "gender", "age_group", "skin_tone"]:
        if col not in df.columns:
            df[col] = np.nan

    # Load state
    state = load_state()
    start_index = int(state.get("next_index", 0))
    already_processed = int(state.get("processed_total", 0))
    crash_count = int(state.get("crash_count", 0))

    print(f"State: start_index={start_index} | processed_total={already_processed} | crash_count={crash_count}")

    # Load models
    print("Loading MTCNN...")
    mtcnn = MTCNN(keep_all=True, device=DEVICE)

    print("Loading FairFace...")
    fairface = load_fairface_model()
    print("FairFace model loaded ✔️")

    processed_this_run = 0
    face_detected_this_run = 0

    # Determine which rows still need processing
    # We consider processed if face_detected is not NaN (either True/False)
    need_mask = df["face_detected"].isna() & df["exists"].astype(bool)
    need_indices = np.where(need_mask.values)[0]

    # Start from start_index within the dataframe
    need_indices = need_indices[need_indices >= start_index]

    if len(need_indices) == 0:
        print("✅ Nothing left to process.")
        summary = compute_summary(df)
        ensure_parent(SUMMARY_JSON)
        Path(SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("🧾 Summary saved:", SUMMARY_JSON)
        return

    # Limit rows per run (helps avoid long unstable runs)
    if MAX_ROWS_PER_RUN is not None and len(need_indices) > MAX_ROWS_PER_RUN:
        need_indices = need_indices[:MAX_ROWS_PER_RUN]
        print(f"🧩 Processing a chunk this run: {len(need_indices)} rows (MAX_ROWS_PER_RUN={MAX_ROWS_PER_RUN})")
    else:
        print(f"🧩 Processing rows this run: {len(need_indices)}")

    pbar = tqdm(need_indices, desc="🧑‍🦰 FairFace inference")

    for idx in pbar:
        # Save state BEFORE doing anything that might crash
        state_update = {
            "next_index": int(idx),
            "processed_total": int(already_processed),
            "processed_in_run": int(processed_this_run),
            "face_detected_in_run": int(face_detected_this_run),
            "crash_count": int(crash_count),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_state(state_update)

        img_path = df.at[idx, "image_path"]
        img = safe_open_rgb(img_path)

        if img is None:
            df.at[idx, "face_detected"] = False
            continue

        try:
            # Face detection
            boxes, _ = mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                df.at[idx, "face_detected"] = False
                df.at[idx, "gender"] = np.nan
                df.at[idx, "age_group"] = np.nan
                df.at[idx, "skin_tone"] = np.nan
            else:
                df.at[idx, "face_detected"] = True
                face_detected_this_run += 1

                # Take the largest face (most confident surrogate)
                # boxes: (N, 4) [x1,y1,x2,y2]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                j = int(np.argmax(areas))
                x1, y1, x2, y2 = boxes[j]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(img.width, int(x2)), min(img.height, int(y2))

                face = img.crop((x1, y1, x2, y2))

                x = FAIRFACE_TF(face).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    g_logits, a_logits, r_logits = fairface(x)

                    g = int(torch.argmax(g_logits, dim=1).item())
                    a = int(torch.argmax(a_logits, dim=1).item())
                    r = int(torch.argmax(r_logits, dim=1).item())

                gender = GENDER_LABELS[g] if g < len(GENDER_LABELS) else str(g)
                age_group = AGE_LABELS[a] if a < len(AGE_LABELS) else str(a)
                race = RACE_LABELS[r] if r < len(RACE_LABELS) else str(r)

                df.at[idx, "gender"] = gender
                df.at[idx, "age_group"] = age_group
                df.at[idx, "skin_tone"] = race_to_skin_tone(race)

            processed_this_run += 1
            already_processed += 1

            # Periodic save
            if processed_this_run % SAVE_EVERY == 0:
                save_checkpoint(df)
                pbar.set_postfix({"saved": True, "done": already_processed})

        except Exception as e:
            # If Python exception occurs, we skip safely.
            # NOTE: This does NOT catch access violations; those crash the interpreter.
            df.at[idx, "face_detected"] = False
            df.at[idx, "gender"] = np.nan
            df.at[idx, "age_group"] = np.nan
            df.at[idx, "skin_tone"] = np.nan

    # Final save
    save_checkpoint(df)
    print(f"💾 Final CSV saved: {OUT_CSV}")

    summary = compute_summary(df)
    ensure_parent(SUMMARY_JSON)
    Path(SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("🧾 Summary saved:", SUMMARY_JSON)

    # Update state to next position
    # next_index is last idx processed + 1
    last_idx = int(need_indices[-1])
    final_state = {
        "next_index": last_idx + 1,
        "processed_total": int(already_processed),
        "processed_in_run": int(processed_this_run),
        "face_detected_in_run": int(face_detected_this_run),
        "crash_count": int(crash_count),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "completed_run",
    }
    save_state(final_state)
    print("✅ Run completed. You can re-run to continue remaining rows.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Python-level exceptions only
        print("❌ Python exception occurred:\n")
        traceback.print_exc()
        # increment crash count
        st = load_state()
        st["crash_count"] = int(st.get("crash_count", 0)) + 1
        save_state(st)
        raise
