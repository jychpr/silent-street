"""
FastSAM RANKING DIAGNOSTIC — inspection only, no fixes, no metric changes.

Question: the K=5 cap discards ~94% of geometrically-valid FastSAM boxes.
Are the 5 that survive the RIGHT 5, or are novel objects ranked below stuff and
cut by the cap?

Pipeline replicated EXACTLY from tools/precompute_fastsam_pseudo_labels.py:
  FastSAM-x, conf=0.10, iou=0.7, imgsz=1024, retina_masks=True, native box.xyxy
  geometric filter (min_side=4, max_area_ratio=0.4, max_aspect=5.0)
  THEN sort-by-conf desc, THEN keep top-5.

The saved _filtered.json only contains the post-cap top-5, so FastSAM is re-run
here to recover the PRE-CAP survivors. Faithfulness is asserted by checking the
recomputed top-5 reproduces the saved top-5 per image.

STOP after the report. No logic changes anywhere.
"""

import json
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np

# ---- exact settings (confirmed/log-verified) --------------------------------
DEVICE = "cuda:0"
WEIGHTS = "FastSAM-x.pt"
CONF, IOU, IMGSZ = 0.10, 0.7, 1024
MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT = 4.0, 0.4, 5.0
TOPK = 5
IOU_MATCH = 0.5

IMAGES = [579329, 122263, 157105, 443084, 92869, 475808, 34]

NOVEL = {"airplane", "bus", "cat", "dog", "cow", "elephant", "umbrella", "tie",
         "snowboard", "skateboard", "cup", "knife", "cake", "couch", "keyboard",
         "sink", "scissors"}

FULL_GT = "data/Annotations/instances_train2017.json"
BASE_GT = "data/Annotations/instances_train2017_12img_diag.json"  # 48 base names
FILT = "ow_labels/OW_COCO_FASTSAM_K5_conf010_filtered.json"
WCONF = "ow_labels/OW_COCO_FASTSAM_K5_conf010_wconf.json"


def iou_1ton(b, B):
    """IoU of one xyxy box b:(4,) against B:(M,4) xyxy. Returns (M,)."""
    if len(B) == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    iw = np.clip(x2 - x1, 0, None); ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    ab = (b[2] - b[0]) * (b[3] - b[1])
    aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = ab + aB - inter
    return np.where(union > 0, inter / union, 0.0)


def gpu_guard():
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().strip().splitlines()[0]
    procs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    ).decode().strip()
    print(f"GPU pre-run: {used} MiB used, compute-procs=[{procs or 'none'}]")
    if int(used) > 2000 or procs:
        raise SystemExit("ABORT (Rule 7): GPU not free.")


def load_base_names():
    d = json.load(open(BASE_GT))
    return {c["name"] for c in d["categories"]}


def tier_of(name, base):
    if name in NOVEL:
        return "novel"
    if name in base:
        return "base"
    return "other"


def load_gt(base):
    print(f"Loading full GT {FULL_GT} (~big, one-time) ...")
    t = time.time()
    d = json.load(open(FULL_GT))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    wh = {im["id"]: (im["width"], im["height"]) for im in d["images"] if im["id"] in IMAGES}
    gt = defaultdict(list)  # iid -> list of (xyxy np, name, tier)
    want = set(IMAGES)
    for a in d["annotations"]:
        iid = a["image_id"]
        if iid not in want:
            continue
        x, y, bw, bh = a["bbox"]
        name = cats[a["category_id"]]
        gt[iid].append((np.array([x, y, x + bw, y + bh], dtype=np.float64),
                        name, tier_of(name, base)))
    print(f"  GT loaded in {time.time()-t:.0f}s")
    return gt, wh


def load_saved():
    fb = defaultdict(list)
    wconf = defaultdict(list)
    want = set(IMAGES)
    for a in json.load(open(FILT)):
        if a["image_id"] in want:
            fb[a["image_id"]].append(tuple(round(v, 4) for v in a["bbox"]))
    for a in json.load(open(WCONF)):
        if a["image_id"] in want:
            wconf[a["image_id"]].append(a["weight"])
    return fb, wconf


def survivors_for(model, img_path, iw, ih):
    """Re-run FastSAM, apply geometric filter, return survivors sorted by conf desc."""
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, verbose=False)
    if not res or res[0].boxes is None:
        return []
    image_area = iw * ih
    cand = []
    for box in res[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        c = float(box.conf[0])
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if bw * bh > MAX_AREA_RATIO * image_area:
            continue
        if max(bw / bh, bh / bw) > MAX_ASPECT:
            continue
        cand.append((x1, y1, x2, y2, c))
    cand.sort(key=lambda t: t[4], reverse=True)
    return cand


def main():
    gpu_guard()
    base = load_base_names()
    print(f"Base classes: {len(base)}  Novel: {len(NOVEL)}  (other = COCO-80 minus these)")
    gt, wh = load_gt(base)
    saved_fb, saved_wconf = load_saved()

    from ultralytics import FastSAM  # deferred
    model = FastSAM(WEIGHTS)
    print(f"FastSAM loaded from {WEIGHTS}\n")

    from pathlib import Path
    img_dir = Path("data/Images/train2017")

    per_image = {}
    novel_rank_hist = []   # best-match rank for each matched novel GT (across all imgs)
    faithful = True

    for iid in IMAGES:
        iw, ih = wh[iid]
        # locate file
        fn = f"{iid:012d}.jpg"
        img_path = str(img_dir / fn)
        if not Path(img_path).exists():
            raise SystemExit(f"missing image {img_path}")

        surv = survivors_for(model, img_path, iw, ih)
        n_surv = len(surv)
        S = np.array([[s[0], s[1], s[2], s[3]] for s in surv], dtype=np.float64) if surv else np.zeros((0, 4))
        confs = [s[4] for s in surv]

        # ---- faithfulness: recomputed top-5 vs saved top-5 ------------------
        mine_top5 = {tuple(round(v, 4) for v in (s[0], s[1], s[2] - s[0], s[3] - s[1]))
                     for s in surv[:TOPK]}
        saved_set = set(saved_fb[iid])
        exact = (mine_top5 == saved_set)
        if not exact:
            faithful = False

        # ---- per-survivor landing (obj vs stuff, tier) ----------------------
        G = np.array([g[0] for g in gt[iid]], dtype=np.float64) if gt[iid] else np.zeros((0, 4))
        g_tier = [g[2] for g in gt[iid]]
        g_name = [g[1] for g in gt[iid]]
        land = []  # per survivor: (best_iou, name, tier) or (iou,'-','stuff')
        for i in range(n_surv):
            if len(G):
                iou = iou_1ton(S[i], G)
                j = int(np.argmax(iou))
                if iou[j] >= IOU_MATCH:
                    land.append((float(iou[j]), g_name[j], g_tier[j]))
                else:
                    land.append((float(iou[j]), "-", "stuff"))
            else:
                land.append((0.0, "-", "stuff"))

        # ---- per-GT matching: best survivor + its rank ----------------------
        gt_match = []  # (name, tier, best_rank(1-based or None), best_iou, kept_top5)
        for k, (gb, name, tier) in enumerate(gt[iid]):
            if n_surv == 0:
                gt_match.append((name, tier, None, 0.0, False))
                continue
            iou = iou_1ton(gb, S)
            j = int(np.argmax(iou))
            if iou[j] >= IOU_MATCH:
                matching = np.where(iou >= IOU_MATCH)[0]
                kept = bool(matching.min() < TOPK)
                gt_match.append((name, tier, j + 1, float(iou[j]), kept))
                if tier == "novel":
                    novel_rank_hist.append(j + 1)
            else:
                gt_match.append((name, tier, None, float(iou[j]), False))

        per_image[iid] = dict(n_surv=n_surv, confs=confs, exact=exact,
                              mine_top5=mine_top5, saved_set=saved_set,
                              land=land, gt_match=gt_match)

        # ---------- per-image report ----------------------------------------
        print("=" * 78)
        print(f"IMAGE {iid}  ({iw}x{ih})   survivors(pre-cap)={n_surv}   "
              f"cap keeps top-{min(TOPK,n_surv)}  -> discards {max(0,n_surv-TOPK)}")
        gtc = Counter((t, n) for n, t in [(g[1], g[2]) for g in gt[iid]])
        nov = sum(1 for g in gt[iid] if g[2] == "novel")
        bas = sum(1 for g in gt[iid] if g[2] == "base")
        oth = sum(1 for g in gt[iid] if g[2] == "other")
        print(f"  GT: novel={nov} base={bas} other={oth}  "
              f"[{' '.join(f'{n}*:{c}' if t==chr(110)+'ovel' else f'{n}:{c}' for (t,n),c in sorted(gtc.items()))}]")
        print(f"  faithfulness (recomputed top-5 == saved top-5): {'OK' if exact else 'MISMATCH'}")

        print("  TOP-5 KEPT (rank: conf  -> lands on):")
        for r in range(min(TOPK, n_surv)):
            bi, nm, tr = land[r]
            tag = f"{nm}({tr}) iou={bi:.2f}" if nm != "-" else f"STUFF/bg (best iou={bi:.2f})"
            star = " <NOVEL" if tr == "novel" else ""
            print(f"    {r+1}: {confs[r]:.3f}  -> {tag}{star}")

        # discarded survivors obj vs stuff
        disc = land[TOPK:]
        d_obj = sum(1 for bi, nm, tr in disc if nm != "-")
        d_nov = sum(1 for bi, nm, tr in disc if tr == "novel")
        d_bas = sum(1 for bi, nm, tr in disc if tr == "base")
        d_oth = sum(1 for bi, nm, tr in disc if tr == "other")
        d_stuff = len(disc) - d_obj
        print(f"  DISCARDED (rank 6+, n={len(disc)}): land-on-object={d_obj} "
              f"(novel={d_nov} base={d_bas} other={d_oth})  stuff/bg={d_stuff}")

        # novel & base matched any-rank vs top-5
        def cnt(tier):
            anyr = sum(1 for n, t, rk, io, kp in gt_match if t == tier and rk is not None)
            t5 = sum(1 for n, t, rk, io, kp in gt_match if t == tier and kp)
            tot = sum(1 for g in gt[iid] if g[2] == tier)
            return tot, anyr, t5
        nv_tot, nv_any, nv_t5 = cnt("novel")
        bs_tot, bs_any, bs_t5 = cnt("base")
        print(f"  NOVEL GT: total={nv_tot}  matched-by-a-survivor(any rank)={nv_any}  "
              f"kept-in-top5={nv_t5}   >> CAP COST = {nv_any - nv_t5}")
        print(f"  BASE  GT: total={bs_tot}  matched(any rank)={bs_any}  kept-in-top5={bs_t5}   "
              f"cap cost = {bs_any - bs_t5}")

        # novel GT detail (the smoking gun)
        nv_rows = [(n, rk, io, kp) for n, t, rk, io, kp in gt_match if t == "novel" and rk is not None]
        if nv_rows:
            print("  novel GT matched by a survivor (name: best-rank, iou, kept?):")
            for n, rk, io, kp in sorted(nv_rows, key=lambda x: (x[1] is None, x[1])):
                print(f"      {n}: rank={rk} iou={io:.2f} {'KEPT' if kp else 'CUT by cap'}")

    # ---------------- special drill-downs --------------------------------
    print("\n" + "#" * 78)
    print("SPECIAL DRILL-DOWNS")
    # 579329 persons (grandpa)
    print("\n[579329] persons (base) — ranks:")
    gm = per_image[579329]["gt_match"]
    persons = [(n, rk, io, kp) for n, t, rk, io, kp in gm if n == "person"]
    # largest person by GT area = likely 'grandpa'
    areas = [( (g[0][2]-g[0][0])*(g[0][3]-g[0][1]) ) for g in gt[579329] if g[1]=="person"]
    print(f"  {len(persons)} person GT; "
          f"matched(any)={sum(1 for _,rk,_,_ in persons if rk)} kept-top5={sum(1 for _,_,_,kp in persons if kp)}")
    for n, rk, io, kp in sorted(persons, key=lambda x: (x[1] is None, x[1])):
        print(f"    person: rank={rk} iou={io:.2f} {'KEPT' if kp else ('CUT' if rk else 'NOT-MATCHED@0.5')}")

    # 92869 dogs + umbrellas (novel)
    print("\n[92869] dogs + umbrellas (novel) — ranks:")
    gm = per_image[92869]["gt_match"]
    for cls in ("dog", "umbrella"):
        rows = [(rk, io, kp) for n, t, rk, io, kp in gm if n == cls]
        tot = len(rows)
        anyr = sum(1 for rk, io, kp in rows if rk)
        t5 = sum(1 for rk, io, kp in rows if kp)
        ranks = sorted([rk for rk, io, kp in rows if rk])
        print(f"  {cls}: total={tot} matched(any)={anyr} kept-top5={t5}  "
              f"ranks-of-matched={ranks}")

    # ---------------- aggregate novel rank distribution ------------------
    print("\n" + "#" * 78)
    print("AGGREGATE — rank distribution of novel GT matched by a survivor (best-match rank):")
    buckets = {"1-5 (kept)": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51+": 0}
    for r in novel_rank_hist:
        if r <= 5: buckets["1-5 (kept)"] += 1
        elif r <= 10: buckets["6-10"] += 1
        elif r <= 20: buckets["11-20"] += 1
        elif r <= 50: buckets["21-50"] += 1
        else: buckets["51+"] += 1
    total_nv_matched = len(novel_rank_hist)
    kept = buckets["1-5 (kept)"]
    print(f"  novel GT matched by some survivor: {total_nv_matched}")
    for k, v in buckets.items():
        print(f"    rank {k:12s}: {v}")
    print(f"  >> novel objects FastSAM FOUND but the K=5 cap CUT: {total_nv_matched - kept} "
          f"of {total_nv_matched}")

    print("\n" + "#" * 78)
    print(f"FAITHFULNESS: recomputed top-5 reproduces saved _filtered.json for all images: "
          f"{'YES' if faithful else 'NO — see MISMATCH above'}")
    print("STOP — inspection only, no metric changes.")


if __name__ == "__main__":
    main()
