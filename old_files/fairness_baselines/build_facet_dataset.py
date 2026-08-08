#!/usr/bin/env python
# fairness/build_facet_dataset.py
"""
Create a FACET evaluation CSV with:
    - hard skin tone label (optional, with tie strategies),
    - soft probabilities over tones (1..10).

It expects an annotations CSV with columns including:
    filename, bounding_box (JSON with x,y,width,height), and skin_tone_1..skin_tone_10 (vote counts).

Example:
python fairness/build_facet_dataset.py \
    --ann_csv facet_data/annotations/annotations.csv \
    --img_dirs facet_data/imgs_1 facet_data/imgs_2 \
    --out_csv facet_data/facet_eval.csv \
    --hard_strategy median_round \
    --visible_face_col visible_face
"""
#!/usr/bin/env python
# fairness/build_facet_dataset_plus.py
import argparse, os, json, random
import pandas as pd
from pathlib import Path

def weighted_median(values, weights):
    pairs = sorted(zip(values, weights))
    total = sum(weights); cum = 0
    for v, w in pairs:
        cum += w
        if cum >= 0.5 * total:
            return v
    return pairs[-1][0]

def choose_hard_label(counts, strategy="median_round", seed=42):
    tones = list(range(1, 11))
    mx = max(counts); ties = [t for t, c in zip(tones, counts) if c == mx]
    if len(ties) == 1: return ties[0]
    if strategy == "mode_high": return max(ties)
    if strategy == "mode_low":  return min(ties)
    if strategy == "mode_random":
        import random as _r; _r.seed(seed); return _r.choice(ties)
    if strategy == "drop_ties": return None
    if strategy == "median_round": return int(round(weighted_median(tones, counts)))
    return max(ties)

def _to_bool01(v):
    if pd.isna(v): return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","yes","y","t"}: return 1
        if s in {"0","false","no","n","f"}: return 0
        try:
            fv = float(s); return 1 if fv != 0 else 0
        except: return None
    if isinstance(v, (int, float)): return 1 if v != 0 else 0
    if isinstance(v, bool): return 1 if v else 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="facet_data")
    ap.add_argument("--ann_csv", required=True, help="Raw FACET annotations CSV")
    ap.add_argument("--img_dirs", nargs="+", required=True)
    ap.add_argument("--out_csv", default="facet_data/facet_eval.csv")
    ap.add_argument("--hard_strategy", default="median_round",
        choices=["median_round","mode_high","mode_low","mode_random","drop_ties","none"])
    ap.add_argument("--visible_face_col", default="visible_face",
        help="Column name in annotations for face visibility (e.g., 'visible_face' or 'face_visible').")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.ann_csv)

    # detect skin tone columns
    skin_cols = [c for c in df.columns if c.startswith("skin_tone_") and c not in ("skin_tone_na",)]
    df = df[df[skin_cols].sum(axis=1) > 0].reset_index(drop=True)

    # parse bbox (JSON in column 'bounding_box')
    bbox = df["bounding_box"].apply(json.loads)
    df["x"] = bbox.apply(lambda b: b["x"])
    df["y"] = bbox.apply(lambda b: b["y"])
    df["width"]  = bbox.apply(lambda b: b["width"])
    df["height"] = bbox.apply(lambda b: b["height"])

    # attach image paths (by filename)
    path_map = {}
    for d in args.img_dirs:
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                path_map[fn] = os.path.join(d, fn)
    df["image_path"] = df["filename"].map(path_map)

    # hard+soft labels
    hard_labels, soft_probs = [], []
    for _, row in df.iterrows():
        counts = [int(row.get(f"skin_tone_{i}", 0)) for i in range(1, 11)]
        tot = sum(counts)
        if tot <= 0:
            hard_labels.append(None); soft_probs.append(json.dumps([0.0]*10)); continue
        probs = [c/tot for c in counts]
        soft_probs.append(json.dumps(probs))
        hard = None if args.hard_strategy=="none" else choose_hard_label(counts, args.hard_strategy, seed=args.seed)
        hard_labels.append(hard)

    # visible_face (optional)
    vis_col = args.visible_face_col if args.visible_face_col in df.columns else None
    if vis_col is not None:
        vis_vals = df[vis_col].apply(_to_bool01)
    else:
        vis_vals = None

    df_out = pd.DataFrame({
        "image_path": df["image_path"],
        "filename": df["filename"],
        "x": df["x"], "y": df["y"], "width": df["width"], "height": df["height"],
        "skin_tone_final": hard_labels,
        "skin_tone_probs": soft_probs,
    })
    if vis_vals is not None:
        df_out["visible_face"] = vis_vals

    # drop rows with missing images
    df_out = df_out[df_out["image_path"].notna()]
    if args.hard_strategy == "drop_ties":
        df_out = df_out[df_out["skin_tone_final"].notna()]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"→ Wrote {len(df_out)} rows to {args.out_csv} (visible_face={'kept' if vis_vals is not None else 'absent'})")

if __name__ == "__main__":
    main()
