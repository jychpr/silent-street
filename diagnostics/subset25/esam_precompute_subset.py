"""
esam_precompute_subset.py — serialize E-SAM final-mask boxes into OLN-schema
pseudo-label records for the 25% subset, so E-SAM priors can feed the IDENTICAL
OV-DQUO training path the OLN priors use.

Per subset image_id:
  load image -> run_esam (esam_pipeline; full MMG->EMR->USR, equation code
                          imported verbatim) -> final_masks
  each final mask -> mask_to_box (gate_d_eval, returns xyxy) -> passes_geo
                     (gate_d_eval geometric filter, VERBATIM) -> convert to xywh
                  -> OLN-schema record:
        {bbox:[x,y,w,h], image_id, id, category_id:-1, pseudo:1, weight:1.0}

  weight is UNIFORM 1.0 BY DESIGN: E-SAM carries no objectness/foreground score,
  unlike OLN-R2's learned weight. The loader applies weight**0.5 as a per-box
  classification-loss multiplier; uniform 1.0 is the documented "trust all boxes
  equally" choice. Masks are NOT persisted — boxes only.

Resumable: per-image done-markers in a work dir; completed ids are skipped on
restart; records are appended to JSONL incrementally and merged into one
OLN-schema list at the end. All progress prints to stdout (foreground).

FULL run (hand-run by human — NOT auto-launched):
  python diagnostics/subset25/esam_precompute_subset.py
SMOKE (first N images only; separate work dir + output; safe to run):
  python diagnostics/subset25/esam_precompute_subset.py --smoke 5
"""
import argparse
import json
import os
import sys
import time

import cv2
import torch

DDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # diagnostics/
sys.path.insert(0, DDIR)
from esam_pipeline import run_esam                       # noqa: E402  full E-SAM, verbatim
from gate_d_eval import mask_to_box, passes_geo, ROOT    # noqa: E402  box+geo, verbatim

DEVICE = "cuda:0"                                  # esam_pipeline is hardcoded to cuda:0
SAM_CKPT = f"{ROOT}/weights/sam_vit_h_4b8939.pth"
IDS_JSON = f"{ROOT}/diagnostics/subset25/subset_25pct_ids.json"
IMG_DIR = f"{ROOT}/data/Images/train2017"
ESAM_ID_BASE = 96_000_000_000                      # namespace for sequential pseudo ids


def gpu_safety_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — abort.")
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {(total - free) / 1e9:.1f} GB used / {total / 1e9:.1f} GB total "
          f"({free / 1e9:.1f} GB free)")
    if free < 6e9:
        raise RuntimeError(f"Only {free / 1e9:.1f} GB free — GPU busy, abort before inference.")


def load_predictor():
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam.to(DEVICE).eval()
    print("Loaded SAM-H")
    return SamPredictor(sam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="process only the first N subset images into a separate smoke output")
    args = ap.parse_args()

    all_ids = json.load(open(IDS_JSON))["image_ids"]
    if args.smoke > 0:
        ids = all_ids[: args.smoke]
        work = f"{ROOT}/diagnostics/subset25/esam_work_smoke"
        out = f"{ROOT}/ow_labels/OW_COCO_ESAM_uniform_subset25_smoke{args.smoke}.json"
        tag = f"SMOKE({args.smoke})"
    else:
        ids = all_ids
        work = f"{ROOT}/diagnostics/subset25/esam_work"
        out = f"{ROOT}/ow_labels/OW_COCO_ESAM_uniform_subset25.json"
        tag = "FULL"
    os.makedirs(work, exist_ok=True)
    rec_path = f"{work}/records.jsonl"
    done_path = f"{work}/done.jsonl"
    err_path = f"{work}/errors.jsonl"

    # resume: completed image ids
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            line = line.strip()
            if line:
                done.add(json.loads(line)["image_id"])
    todo = [i for i in ids if i not in done]
    print(f"[{tag}] subset={len(ids)}  already_done={len(done & set(ids))}  "
          f"todo={len(todo)}  work={work}")

    if todo:
        gpu_safety_check()
        predictor = load_predictor()
        print(f"Geo filter reused verbatim from gate_d_eval "
              f"(min_side=4, max_area_ratio=0.4, max_aspect=5.0)")
        print("=" * 80)
        rec_f = open(rec_path, "a")
        done_f = open(done_path, "a")
        err_f = open(err_path, "a")
        t_start = time.perf_counter()
        n_err = 0
        for k, iid in enumerate(todo, 1):
            path = f"{IMG_DIR}/{iid:012d}.jpg"
            nb = 0
            try:
                bgr = cv2.imread(path)
                if bgr is None:
                    raise FileNotFoundError(path)
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                H, W = img_rgb.shape[:2]
                res = run_esam(predictor, img_rgb, H, W)
                for m in res["final_masks"]:
                    b = mask_to_box(m)                       # xyxy or None
                    if b is None or not passes_geo(b, W, H):
                        continue
                    x, y, x2, y2 = b
                    rec_f.write(json.dumps({
                        "bbox": [x, y, x2 - x, y2 - y],      # xyxy -> xywh (OLN schema)
                        "image_id": iid, "category_id": -1,
                        "pseudo": 1, "weight": 1.0}) + "\n")
                    nb += 1
                rec_f.flush()
                done_f.write(json.dumps({"image_id": iid, "n_boxes": nb}) + "\n")
                done_f.flush()
            except Exception as e:                            # noqa: BLE001 — long unattended run
                n_err += 1
                err_f.write(json.dumps({"image_id": iid, "error": repr(e)}) + "\n")
                err_f.flush()
                done_f.write(json.dumps({"image_id": iid, "n_boxes": 0, "error": True}) + "\n")
                done_f.flush()
                print(f"  !! {iid}: ERROR {e!r} (recorded, skipping)")
                continue

            if k == 1 or k % 100 == 0 or k == len(todo):
                el = time.perf_counter() - t_start
                rate = el / k
                eta = rate * (len(todo) - k)
                print(f"  [{k}/{len(todo)}] id={iid} boxes={nb}  "
                      f"elapsed={el / 60:.1f}m  rate={rate:.2f}s/img  "
                      f"ETA={eta / 60:.1f}m ({eta / 3600:.2f}h)")
        rec_f.close(); done_f.close(); err_f.close()
        print("=" * 80)
        print(f"[{tag}] inference pass complete. errors={n_err}")

    # merge -> single OLN-schema list with sequential unique ids
    recs = []
    if os.path.exists(rec_path):
        for line in open(rec_path):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    for seq, r in enumerate(recs):
        r["id"] = ESAM_ID_BASE + seq
    json.dump(recs, open(out, "w"))

    done_ids = set()
    for line in open(done_path):
        done_ids.add(json.loads(line)["image_id"])
    processed = len(done_ids & set(ids))
    complete = processed >= len(ids)
    imgs = {r["image_id"] for r in recs}
    print(f"[{tag}] merged {len(recs)} records over {len(imgs)} images -> {out}")
    if recs:
        print(f"  sample record: {recs[0]}")
        print(f"  record keys  : {sorted(recs[0].keys())}")
    print(f"STATUS: {'COMPLETE' if complete else 'PARTIAL — rerun to resume'} "
          f"({processed}/{len(ids)} images processed)")


if __name__ == "__main__":
    main()
