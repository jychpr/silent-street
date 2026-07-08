"""
make_oln_subset.py — filter the OLN prior (ow_labels/OW_COCO_R2.json) to records
whose image_id is in the 25% subset. Writes ow_labels/OW_COCO_R2_subset25.json
with the IDENTICAL flat-list-of-records schema (subsettable by image_id as-is,
no re-aggregation).

Run (after make_subset_ids.py):
  python diagnostics/subset25/make_oln_subset.py
"""
import json

ROOT = "/home/akihito/JC/lab/silent-street"
IDS = f"{ROOT}/diagnostics/subset25/subset_25pct_ids.json"
R2 = f"{ROOT}/ow_labels/OW_COCO_R2.json"
OUT = f"{ROOT}/ow_labels/OW_COCO_R2_subset25.json"


def main():
    ids = set(json.load(open(IDS))["image_ids"])
    print(f"subset images: {len(ids)}")
    print(f"loading {R2} ...")
    recs = json.load(open(R2))
    kept = [r for r in recs if r["image_id"] in ids]
    json.dump(kept, open(OUT, "w"))

    imgs_with = {r["image_id"] for r in kept}
    n_with = len(imgs_with)
    n_zero = len(ids) - n_with
    n_boxes = len(kept)

    print(f"wrote {OUT}")
    print(f"  total boxes                 : {n_boxes}")
    print(f"  subset images with >=1 OLN  : {n_with}")
    print(f"  subset images with 0 OLN    : {n_zero}")
    print(f"  mean boxes / subset image   : {n_boxes / len(ids):.3f}   (over all {len(ids)})")
    print(f"  mean boxes / covered image  : {n_boxes / max(n_with, 1):.3f}   (over {n_with} covered)")
    if kept:
        print(f"  record keys                 : {sorted(kept[0].keys())}")
        print(f"  sample record               : {kept[0]}")


if __name__ == "__main__":
    main()
