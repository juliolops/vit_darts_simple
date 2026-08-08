#!/usr/bin/env python
"""
download_wider_places.py  —  robust downloader/arranger

This version avoids Google Drive quota by preferring HTTP mirrors and
correctly stages Places365 'val' even when extracted as 'val_large' or 'val_256'.

Outputs to <repo_root>/data by default (assuming this file lives at
<repo_root>/fairness/download_wider_places.py). Override with --out_root.

Usage:
    python fairness/download_wider_places.py
    python fairness/download_wider_places.py --out_root /mnt/fastdisk/data
    python fairness/download_wider_places.py --skip_wider           # if WIDER is already done
    python fairness/download_wider_places.py --places_split train   # if you want train instead of val

Requires:
    pip install --upgrade torchvision pillow tqdm requests
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm

# --- Optional torchvision fallback (kept last) ---
try:
    from torchvision.datasets import WIDERFace, Places365
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# =========================
# Path helpers
# =========================

def repo_root_from_script(expected_parent_name: str = "fairness") -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent  # robust even if the folder isn't named exactly "fairness"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def merge_or_move(src: Path, dst: Path) -> None:
    """Move directory tree from src to dst. If dst exists, merge non-destructively."""
    if not src.exists():
        return
    ensure_dir(dst)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        cur_dst = dst / rel
        ensure_dir(cur_dst)
        for d in dirs:
            ensure_dir(cur_dst / d)
        for f in files:
            s = Path(root) / f
            d = cur_dst / f
            if not d.exists():
                try:
                    shutil.move(str(s), str(d))
                except Exception:
                    shutil.copy2(str(s), str(d))
    shutil.rmtree(src, ignore_errors=True)


def find_subdir(root: Path, name: str) -> Optional[Path]:
    candidates = [p for p in root.rglob(name) if p.is_dir()]
    return candidates[0] if candidates else None


def count_images(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*.jpg")) + \
        sum(1 for _ in folder.rglob("*.jpeg")) + \
        sum(1 for _ in folder.rglob("*.png"))


# =========================
# HTTP download helpers
# =========================

def http_download(url: str, dst: Path) -> None:
    """Stream download with progress; raises on HTTP errors."""
    ensure_dir(dst.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def unzip_to(src_zip: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    with zipfile.ZipFile(src_zip, "r") as z:
        z.extractall(dst_dir)


# =========================
# WIDER download strategies
# =========================

def download_wider_huggingface(tmp_root: Path) -> bool:
    """Hugging Face mirror (no Google Drive)."""
    try:
        urls = {
            "WIDER_train.zip": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
            "WIDER_val.zip":   "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
            "wider_face_split.zip": "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip",
        }
        for name, url in urls.items():
            zip_path = tmp_root / name
            if not zip_path.exists():
                http_download(url, zip_path)
            unzip_to(zip_path, tmp_root)
        return True
    except Exception as e:
        print(f"HF mirror failed: {e}")
        return False


def download_wider_mmlab(tmp_root: Path) -> bool:
    """Official CUHK MMLab HTTP links (no Drive)."""
    try:
        urls = {
            "WIDER_train.zip": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WIDER_train.zip",
            "WIDER_val.zip":   "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WIDER_val.zip",
            "wider_face_split.zip": "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip",
        }
        for name, url in urls.items():
            zip_path = tmp_root / name
            if not zip_path.exists():
                http_download(url, zip_path)
            unzip_to(zip_path, tmp_root)
        return True
    except Exception as e:
        print(f"MMLab HTTP failed: {e}")
        return False


def download_wider_torchvision(tmp_root: Path) -> bool:
    """Torchvision fallback (uses Google Drive via gdown and may hit quotas)."""
    if not _HAS_TORCHVISION:
        print("Torchvision not available; skipping torchvision fallback.")
        return False
    try:
        for split in ("train", "val"):
            _ = WIDERFace(root=str(tmp_root), split=split, download=True)
        return True
    except Exception as e:
        print(f"Torchvision WIDERFace download failed: {e}")
        return False


def stage_wider(tmp_root: Path, dst_root: Path) -> None:
    """Move extracted WIDER dirs into canonical layout under data/WIDER/."""
    wider_dst = dst_root / "WIDER"
    ensure_dir(wider_dst)
    wider_train = find_subdir(tmp_root, "WIDER_train")
    wider_val   = find_subdir(tmp_root, "WIDER_val")
    split_dir   = find_subdir(tmp_root, "wider_face_split")
    if wider_train:
        merge_or_move(wider_train, wider_dst / "WIDER_train")
    if wider_val:
        merge_or_move(wider_val, wider_dst / "WIDER_val")
    if split_dir:
        merge_or_move(split_dir, wider_dst / "wider_face_split")


# =========================
# Places365 (torchvision)
# =========================

def download_places(tmp_root: Path, split: str = "val", small: bool = False) -> bool:
    if not _HAS_TORCHVISION:
        print("Torchvision not available; cannot download Places365.")
        return False
    try:
        _ = Places365(root=str(tmp_root), split=split, small=small, download=True)
        return True
    except Exception as e:
        print(f"Places365 download failed: {e}")
        return False


def _candidate_places_dirs(tmp_root: Path, split: str) -> List[Path]:
    """
    Return possible extraction directories for a given split.
    Torchvision often extracts val into 'val_large' or 'val_256'.
    """
    candidates = []
    # 1) exact match
    p = tmp_root / split
    if p.exists():
        candidates.append(p)
    # 2) common variants
    for name in (f"{split}_large", f"{split}_256"):
        q = tmp_root / name
        if q.exists():
            candidates.append(q)
    # 3) recursive search for dirs with lots of images
    for q in tmp_root.iterdir():
        if q.is_dir() and count_images(q) > 1000:  # heuristic: val has many images
            candidates.append(q)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))
    return uniq


def stage_places(tmp_root: Path, dst_root: Path, split: str = "val") -> None:
    """
    Move Places directories into: dst_root/PLACES365/<split>/
    Accepts 'val', 'val_large', 'val_256' (and auto-detects large image dirs).
    """
    places_dst = dst_root / "PLACES365" / split
    candidates = _candidate_places_dirs(tmp_root, split)
    if not candidates:
        print(f"⚠️  Could not find extracted Places365 for split '{split}' under tmp.")
        return
    extracted = candidates[0]
    print(f"   • Found Places at: {extracted}")
    merge_or_move(extracted, places_dst)


# =========================
# Sanity check
# =========================

def sanity_check(data_root: Path) -> bool:
    checks = [
        data_root / "WIDER" / "WIDER_train" / "images",
        data_root / "WIDER" / "WIDER_val" / "images",
        data_root / "WIDER" / "wider_face_split",
        data_root / "PLACES365" / "val",
    ]
    ok = True
    print("\nSanity check:")
    for p in checks:
        if p.name == "wider_face_split":
            exists = p.exists()
            count = sum(1 for _ in p.glob("*.txt")) if exists else 0
        else:
            exists = p.exists()
            count = count_images(p) if exists else 0
        print(f"  • {p}  -> exists={exists}  files={count}")
        ok = ok and exists and count > 0
    return ok


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Download & arrange WIDER FACE + Places365 into a stable layout.")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Custom output root for 'data'. Default: <repo_root>/data inferred from script location.")
    parser.add_argument("--places_split", type=str, default="val", choices=["val", "train"],
                        help="Which Places365 split to download/stage. Default: val (recommended for negatives).")
    parser.add_argument("--places_small", action="store_true",
                        help="With --places_split train, download the 'small' variant (if supported).")
    parser.add_argument("--skip_wider", action="store_true", help="Skip downloading/staging WIDER.")
    parser.add_argument("--skip_places", action="store_true", help="Skip downloading/staging Places365.")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Custom temporary download directory. Default: <out_root>/_tmp_downloads")
    args = parser.parse_args()

    # Resolve repo_root and data root robustly
    repo_root = repo_root_from_script(expected_parent_name="fairness")
    data_root = Path(args.out_root).resolve() if args.out_root else (repo_root / "data")
    ensure_dir(data_root)

    # Temp download root
    tmp_root = Path(args.tmp_dir).resolve() if args.tmp_dir else (data_root / "_tmp_downloads")
    ensure_dir(tmp_root)

    print(f"📁 Repo root: {repo_root}")
    print(f"📦 Data root: {data_root}")
    print(f"🗂️  Temp dir : {tmp_root}")

    wider_tmp = tmp_root / "widerface"
    places_tmp = tmp_root / "places365"
    ensure_dir(wider_tmp); ensure_dir(places_tmp)

    # ---- WIDER FACE ----
    if not args.skip_wider:
        print("\n↓ WIDER FACE: trying Hugging Face mirror...")
        ok = download_wider_huggingface(wider_tmp)
        if not ok:
            print("   Fallback → official CUHK MMLab HTTP...")
            ok = download_wider_mmlab(wider_tmp)
        if not ok:
            print("   Fallback → torchvision (Google Drive via gdown; may hit quota)...")
            ok = download_wider_torchvision(wider_tmp)

        if not ok:
            print("✖ All WIDER download methods failed. "
                "As a last resort, download the three zips manually and unzip into the temp dir:")
            print("  - WIDER_train.zip, WIDER_val.zip (official site)")
            print("  - wider_face_split.zip (annotations)")
            return

        print("→ Staging WIDER into data/WIDER/ ...")
        stage_wider(wider_tmp, data_root)
    else:
        print("\n⏭️  Skipping WIDER (per --skip_wider)")

    # ---- Places365 ----
    if not args.skip_places:
        print(f"\n↓ Places365: downloading split '{args.places_split}' ...")
        okp = download_places(places_tmp, split=args.places_split, small=args.places_small)
        if not okp:
            print("✖ Places365 download failed via torchvision. "
                "Consider manual download from MIT Places website and place under data/PLACES365/<split>/")
        else:
            print(f"→ Staging Places365 '{args.places_split}' into data/PLACES365/{args.places_split}/ ...")
            stage_places(places_tmp, data_root, split=args.places_split)
    else:
        print("\n⏭️  Skipping Places365 (per --skip_places)")

    # Clean temp
    print("\n🧹 Cleaning temporary downloads...")
    shutil.rmtree(tmp_root, ignore_errors=True)

    # Final check
    ok = sanity_check(data_root)
    if ok:
        print("\n✅ All good!")
        print(f"   WIDER at:     {data_root / 'WIDER'}")
        print(f"   PLACES365 at: {data_root / 'PLACES365'}")
        print("\nNext step (example):")
        print("  python fairness/build_face_binary_wider_places.py \\")
        print("    --wider_root  data/WIDER \\")
        print("    --neg_root    data/PLACES365/val \\")
        print("    --out_dir     facebin_data")
    else:
        print("\n⚠️  Something looks off. Re-run with verbose logs or check disk/connection.\n")


if __name__ == "__main__":
    main()
