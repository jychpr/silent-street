"""
make_subset_ids.py — iterative stratified 25% image subset for the controlled
E-SAM-25 vs OLN-25 comparison (same image IDs, same budget).

Pool = images with >=1 base annotation in instances_train2017_base.json
(107,761 images, 48 base categories). Strategy: greedy RAREST-base-category-FIRST
fill toward each category's 25% target, accounting for co-occurrence (an image
selected for a rare class also serves common classes), with a hard floor for the
rarest class (toaster >= 50). Any leftover budget is filled with random pool
images. Fixed seed=42 -> fully deterministic.

Output: diagnostics/subset25/subset_25pct_ids.json
  { seed, n, image_ids:[sorted ints],
    strata_report: { <cat>: {full_count, subset_count, target}, ... } }

Run:
  python diagnostics/subset25/make_subset_ids.py
"""
import json
import os
import random

ROOT = "/home/akihito/JC/lab/silent-street"
BASE_ANN = f"{ROOT}/data/Annotations/instances_train2017_base.json"
OUT = f"{ROOT}/diagnostics/subset25/subset_25pct_ids.json"

SEED = 42
FRACTION = 0.25
FLOORS = {"toaster": 50}   # hard minimum image count for the rarest class


def main():
    rng = random.Random(SEED)
    print(f"Loading {BASE_ANN} ...")
    d = json.load(open(BASE_ANN))
    cats = {c["id"]: c["name"] for c in d["categories"]}

    # per-category image sets; pool = images with >=1 base annotation
    cat_imgs = {cid: set() for cid in cats}
    pool = set()
    for a in d["annotations"]:
        cat_imgs[a["category_id"]].add(a["image_id"])
        pool.add(a["image_id"])
    pool = sorted(pool)
    N = len(pool)
    budget = round(FRACTION * N)
    print(f"pool images (>=1 base ann): {N}   budget (25%): {budget}")

    # per-category target = 25% of its image count, toaster floored
    target = {}
    for cid, name in cats.items():
        t = round(FRACTION * len(cat_imgs[cid]))
        if name in FLOORS:
            t = max(t, FLOORS[name])
        target[cid] = t

    # greedy rarest-first (ascending by full image count); never exceed budget
    order = sorted(cats, key=lambda c: len(cat_imgs[c]))
    selected = set()
    for cid in order:
        if len(selected) >= budget:
            break
        have = len(cat_imgs[cid] & selected)          # co-occurrence carryover
        need = target[cid] - have
        if need <= 0:
            continue
        cand = sorted(cat_imgs[cid] - selected)        # sort -> deterministic shuffle
        rng.shuffle(cand)
        room = budget - len(selected)
        selected.update(cand[: min(need, room)])

    # fill leftover budget with random pool images
    if len(selected) < budget:
        remaining = [iid for iid in pool if iid not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[: budget - len(selected)])

    image_ids = sorted(selected)
    sel = set(image_ids)
    strata = {
        name: {
            "full_count": len(cat_imgs[cid]),
            "subset_count": len(cat_imgs[cid] & sel),
            "target": target[cid],
        }
        for cid, name in cats.items()
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"seed": SEED, "n": len(image_ids), "image_ids": image_ids,
               "strata_report": strata}, open(OUT, "w"))

    # report ascending by full image count
    print(f"\n{'category':22s}{'full':>8s}{'target':>8s}{'subset':>8s}{'frac':>8s}")
    print("-" * 54)
    for name in sorted(strata, key=lambda k: strata[k]["full_count"]):
        r = strata[name]
        frac = r["subset_count"] / r["full_count"] if r["full_count"] else 0.0
        print(f"{name:22s}{r['full_count']:>8d}{r['target']:>8d}"
              f"{r['subset_count']:>8d}{frac:>8.1%}")
    print("-" * 54)
    print(f"\nseed={SEED}   n={len(image_ids)}   (budget {budget}, pool {N})")
    tc = strata["toaster"]["subset_count"]
    print(f"toaster floor (>=50): subset_count={tc}  ->  "
          f"{'MET' if tc >= 50 else 'NOT MET'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
