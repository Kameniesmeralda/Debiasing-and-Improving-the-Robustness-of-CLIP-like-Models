import os
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


# -----------------------------
# Config
# -----------------------------
IN_CSV = "data/laion_big_light_tau1_0.2989.csv"   # 107k light dataset
OUT_CSV = "data/laion_big_light_with_demographics.csv"
SUMMARY_JSON = "data/week5_demographics_summary.json"

CHECKPOINT_EVERY = 200  # save progress every N processed rows
RESUME = True

ACTIONS = ["gender", "age", "race"]
ENFORCE_DETECTION = False

# Optional: pick a backend (mtcnn / retinaface). If you don't specify, DeepFace picks default.
DETECTOR_BACKEND = "mtcnn"   # try "retinaface" if mtcnn is unstable


def find_repo_root() -> Path:
    """Go up until we find a folder that contains 'data/'."""
    p = Path.cwd()
    for _ in range(8):
        if (p / "data").exists():
            return p
        p = p.parent
    return Path.cwd()


def resolve_img_path(root: Path, p: str) -> Path:
    """Resolve relative CSV paths against repo root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def safe_analyze(img_path: str):
    """
    Returns dict with keys: face_detected(bool), gender(str|None), age(int|None), race(str|None)
    """
    try:
        res = DeepFace.analyze(
            img_path=img_path,
            actions=ACTIONS,
            enforce_detection=ENFORCE_DETECTION,
            detector_backend=DETECTOR_BACKEND,
            silent=True,
        )

        if isinstance(res, list):
            res = res[0] if len(res) > 0 else {}

        gender = res.get("dominant_gender", None)
        age = res.get("age", None)
        race = res.get("dominant_race", None)

        face_detected = (gender is not None) or (age is not None) or (race is not None)

        return {
            "face_detected": bool(face_detected),
            "gender": gender,
            "age": age,
            "race": race,
        }

    except Exception:
        return {
            "face_detected": False,
            "gender": None,
            "age": None,
            "race": None,
        }


def main():
    root = find_repo_root()
    in_path = root / IN_CSV
    out_path = root / OUT_CSV
    summary_path = root / SUMMARY_JSON

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume
    if RESUME and out_path.exists():
        print(f"🔁 Resume enabled: loading {out_path}")
        df = pd.read_csv(out_path)
        needed_cols = {"face_detected", "gender", "age", "race"}
        if not needed_cols.issubset(df.columns):
            print("⚠️ Output exists but missing demographic columns. Rebuilding from input...")
            df = pd.read_csv(in_path)
            df["face_detected"] = pd.NA
            df["gender"] = pd.NA
            df["age"] = pd.NA
            df["race"] = pd.NA
    else:
        df = pd.read_csv(in_path)
        df["face_detected"] = pd.NA
        df["gender"] = pd.NA
        df["age"] = pd.NA
        df["race"] = pd.NA

    if "image_path" not in df.columns:
        raise ValueError("CSV must contain an 'image_path' column.")

    need_mask = df["gender"].isna() & df["age"].isna() & df["race"].isna()
    to_process_idx = df.index[need_mask].tolist()
    print(f"📦 Rows: {len(df)} | Remaining to process: {len(to_process_idx)}")

    processed = 0
    detected_faces = int(df["face_detected"].fillna(False).astype(bool).sum())
    start_time = time.time()

    try:
        for idx in tqdm(to_process_idx, desc="🧑 DeepFace demographics"):
            raw_path = df.at[idx, "image_path"]

            if not isinstance(raw_path, str):
                df.at[idx, "face_detected"] = False
                df.at[idx, "gender"] = pd.NA
                df.at[idx, "age"] = pd.NA
                df.at[idx, "race"] = pd.NA
            else:
                p = resolve_img_path(root, raw_path)
                if not p.exists():
                    df.at[idx, "face_detected"] = False
                    df.at[idx, "gender"] = pd.NA
                    df.at[idx, "age"] = pd.NA
                    df.at[idx, "race"] = pd.NA
                else:
                    out = safe_analyze(str(p))
                    df.at[idx, "face_detected"] = out["face_detected"]
                    df.at[idx, "gender"] = out["gender"]
                    df.at[idx, "age"] = out["age"]
                    df.at[idx, "race"] = out["race"]

                    if out["face_detected"]:
                        detected_faces += 1

            processed += 1

            if processed % CHECKPOINT_EVERY == 0:
                df.to_csv(out_path, index=False)
                elapsed = time.time() - start_time
                rate = detected_faces / max(1, (len(df) - len(to_process_idx) + processed))
                print(f"💾 Checkpoint saved | processed={processed} | face_rate≈{rate*100:.1f}% | elapsed={elapsed/60:.1f} min")

    except KeyboardInterrupt:
        print("🛑 Interrupted by user. Saving checkpoint...")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved: {out_path}")
        raise

    except Exception as e:
        print(f"❌ Crash: {e}. Saving checkpoint...")
        df.to_csv(out_path, index=False)
        print(f"💾 Saved: {out_path}")
        raise

    # Final save
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

    face_rate = df["face_detected"].fillna(False).astype(bool).mean()
    summary = {
        "input_csv": IN_CSV,
        "rows": int(len(df)),
        "face_detected_rate": float(face_rate),
        "gender_non_null": int(df["gender"].notna().sum()),
        "age_non_null": int(df["age"].notna().sum()),
        "race_non_null": int(df["race"].notna().sum()),
        "detector_backend": DETECTOR_BACKEND,
        "enforce_detection": ENFORCE_DETECTION,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"🧾 Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
