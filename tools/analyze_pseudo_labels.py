"""Quick distributional analysis of pseudo-label JSON files."""
import argparse
import json
from collections import Counter

import numpy as np


def analyze(path: str) -> None:
    with open(path) as f:
        anns = json.load(f)

    # Per-image counts
    per_image = Counter(a["image_id"] for a in anns)
    counts = np.array(list(per_image.values()))

    # Box areas (xywh)
    areas = np.array([a["bbox"][2] * a["bbox"][3] for a in anns])

    # COCO scale bins
    small = (areas < 32 * 32).sum()
    medium = ((areas >= 32 * 32) & (areas < 96 * 96)).sum()
    large = (areas >= 96 * 96).sum()
    n = len(areas)

    print(f"\n=== {path} ===")
    print(f"Total annotations    : {n:,}")
    print(f"Unique image_ids     : {len(per_image):,}")
    print(f"Boxes/image          : mean={counts.mean():.2f}  "
          f"median={int(np.median(counts))}  "
          f"min={counts.min()}  max={counts.max()}")
    print(f"Boxes/image histogram:")
    for k in sorted(per_image_counts := Counter(counts.tolist())):
        pct = 100 * per_image_counts[k] / len(counts)
        print(f"  {k:>3} boxes: {per_image_counts[k]:>7,} images ({pct:5.2f}%)")
    print(f"Box scale (COCO defn, by area):")
    print(f"  small  (<32²    = <1024 px²) : {small:>9,} ({100*small/n:5.2f}%)")
    print(f"  medium (32²–96² = 1024–9216) : {medium:>9,} ({100*medium/n:5.2f}%)")
    print(f"  large  (>=96²   = >=9216)    : {large:>9,} ({100*large/n:5.2f}%)")
    print(f"Box area percentiles (px²):")
    for p in (1, 10, 25, 50, 75, 90, 99):
        print(f"  p{p:>2}: {int(np.percentile(areas, p)):>8,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        analyze(p)