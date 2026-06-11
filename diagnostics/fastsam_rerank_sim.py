"""
FastSAM RE-RANK SIMULATION — inspection only, no training, no precompute changes.

Tests whether a DIFFERENT selection rule at FIXED K=5 recovers the novel objects
that confidence-sort buries, WITHOUT admitting more stuff. K stays 5 (we are NOT
recreating the Gate D box-flood).

The ranking diagnostic did not persist survivors and never captured mask areas
(R2 needs them). FastSAM is deterministic here (the diagnostic proved the top-5
reproduce exactly), so this script regenerates the IDENTICAL survivor set and
HARD-ASSERTS per-image survivor counts == the diagnostic's, plus conf-sort top-5
== saved _filtered.json. Abort on any mismatch. Then all re-ranking is done on
that fixed survivor set — no re-running per rule.

STOP after the tables. This decides whether a re-rank is worth a training run.
"""

import json
from collections import defaultdict

import numpy as np

from fastsam_rank_diag import (
    NOVEL, IMAGES, DEVICE, WEIGHTS, CONF, IOU, IMGSZ,
    MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT, IOU_MATCH,
    iou_1ton, gpu_guard, load_base_names, load_gt, FILT,
)

# survivor counts the ranking diagnostic reported (faithfulness contract)
EXPECT = {579329: 291, 122263: 292, 157105: 298, 443084: 212, 92869: 139,
          475808: 182, 34: 12}
DENSE = [579329, 122263, 157105, 443084, 92869, 475808]  # zebra 34 has 0 novel
TOPK = 5


def survivors_with_mask(model, img_path, iw, ih):
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, verbose=False)
    r = res[0]
    boxes = r.boxes
    masks = r.masks
    has_mask = (masks is not None and getattr(masks, "data", None) is not None
                and len(masks.data) == len(boxes))
    areas_all = masks.data.sum(dim=(1, 2)).detach().cpu().numpy() if has_mask else None
    image_area = iw * ih
    out = []  # (x1,y1,x2,y2,conf,mask_area)
    for i in range(len(boxes)):
        b = boxes[i]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        c = float(b.conf[0])
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if bw * bh > MAX_AREA_RATIO * image_area:
            continue
        if max(bw / bh, bh / bw) > MAX_ASPECT:
            continue
        marea = float(areas_all[i]) if has_mask else float("nan")
        out.append((x1, y1, x2, y2, c, marea))
    return out, has_mask


def score(top5_idx, S, G, g_tier):
    """box-centric obj/stuff + GT-centric novel/base recovered."""
    obj = 0
    for i in top5_idx:
        if len(G) and float(iou_1ton(S[i], G).max()) >= IOU_MATCH:
            obj += 1
    stuff = len(top5_idx) - obj
    nov = bas = 0
    if len(top5_idx):
        St5 = S[top5_idx]
        for j, tier in enumerate(g_tier):
            m = float(iou_1ton(G[j], St5).max()) if len(St5) else 0.0
            if m >= IOU_MATCH:
                if tier == "novel":
                    nov += 1
                elif tier == "base":
                    bas += 1
    return nov, bas, stuff, obj


def main():
    gpu_guard()
    base = load_base_names()
    gt, wh = load_gt(base)

    # saved top-5 (for faithfulness assert)
    saved_fb = defaultdict(set)
    for a in json.load(open(FILT)):
        if a["image_id"] in set(IMAGES):
            saved_fb[a["image_id"]].add(tuple(round(v, 4) for v in a["bbox"]))

    from ultralytics import FastSAM
    from pathlib import Path
    model = FastSAM(WEIGHTS)
    img_dir = Path("data/Images/train2017")
    print(f"FastSAM loaded from {WEIGHTS}\n")

    imgdata = {}
    mask_ok_all = True
    print("FAITHFULNESS CONTRACT (regenerated survivors must match the diagnostic):")
    for iid in IMAGES:
        iw, ih = wh[iid]
        surv, has_mask = survivors_with_mask(model, str(img_dir / f"{iid:012d}.jpg"), iw, ih)
        # --- assert survivor count matches diagnostic; abort on mismatch ---
        if len(surv) != EXPECT[iid]:
            raise SystemExit(f"ABORT {iid}: survivors={len(surv)} != diagnostic {EXPECT[iid]}")
        surv.sort(key=lambda t: t[4], reverse=True)
        mine_top5 = {tuple(round(v, 4) for v in (s[0], s[1], s[2] - s[0], s[3] - s[1]))
                     for s in surv[:TOPK]}
        if mine_top5 != saved_fb[iid]:
            raise SystemExit(f"ABORT {iid}: recomputed top-5 != saved _filtered.json")
        mask_ok_all &= has_mask

        conf = np.array([s[4] for s in surv])
        S = np.array([[s[0], s[1], s[2], s[3]] for s in surv], dtype=np.float64)
        barea = (S[:, 2] - S[:, 0]) * (S[:, 3] - S[:, 1])
        marea = np.array([s[5] for s in surv])
        G = np.array([g[0] for g in gt[iid]], dtype=np.float64) if gt[iid] else np.zeros((0, 4))
        g_tier = [g[2] for g in gt[iid]]
        imgdata[iid] = dict(conf=conf, S=S, barea=barea, marea=marea,
                            img_area=float(iw * ih), G=G, g_tier=g_tier,
                            n_novel=sum(t == "novel" for t in g_tier))
        print(f"  {iid}: survivors={len(surv)} == {EXPECT[iid]} OK | top-5 OK | masks={'yes' if has_mask else 'NO'}")
    print(f"  ALL faithfulness asserts PASSED. mask areas available: {'yes' if mask_ok_all else 'NO (R2 skipped)'}\n")

    # ---------------- rule engine -----------------------------------------
    def run_rule(keyfn, prefilter=None):
        res = {}
        for iid in IMAGES:
            d = imgdata[iid]
            n = len(d["conf"])
            idx = np.arange(n)
            if prefilter is not None:
                idx = idx[prefilter(d)]
            if len(idx):
                keys = keyfn(d)[idx]
                order = idx[np.argsort(-keys, kind="stable")]
            else:
                order = idx
            top5 = order[:TOPK]
            res[iid] = score(top5, d["S"], d["G"], d["g_tier"])
        return res

    def totals(res, dense_only_novel=True):
        nov = sum(res[i][0] for i in (DENSE if dense_only_novel else IMAGES))
        bas = sum(res[i][1] for i in IMAGES)
        stuff = sum(res[i][2] for i in IMAGES)
        return nov, bas, stuff

    rules = {}
    rules["R0 baseline (conf)"] = run_rule(lambda d: d["conf"])
    rules["R1 area-penalized"] = run_rule(lambda d: d["conf"] * (1 - d["barea"] / d["img_area"]))
    if mask_ok_all:
        rules["R2 compactness (fill)"] = run_rule(lambda d: d["conf"] * (d["marea"] / d["barea"]))
    rules["R3 area-prefilter<=0.15"] = run_rule(
        lambda d: d["conf"], prefilter=lambda d: d["barea"] / d["img_area"] <= 0.15)
    rules["R4 prefilter0.15+areapen"] = run_rule(
        lambda d: d["conf"] * (1 - d["barea"] / d["img_area"]),
        prefilter=lambda d: d["barea"] / d["img_area"] <= 0.15)

    # ---------------- per-rule tables -------------------------------------
    for name, res in rules.items():
        print("=" * 72)
        print(f"RULE {name}")
        print(f"  {'img':>7} | novel_rec | base_rec | stuff_slots | obj_slots")
        for iid in IMAGES:
            nov, bas, stuff, obj = res[iid]
            print(f"  {iid:>7} |    {nov:>2}     |   {bas:>2}     |     {stuff}       |    {obj}")
        nov, bas, stuff = totals(res)
        print(f"  TOTAL   : novel(6 dense)={nov}  base={bas}  stuff_slots={stuff}/35")

    # ---------------- R0 correctness check --------------------------------
    r0 = totals(rules["R0 baseline (conf)"])
    print("\n" + "=" * 72)
    print(f"R0 CONTROL: novel={r0[0]} (expect 4)  stuff={r0[2]} (expect 21)  -> "
          f"{'PASS' if (r0[0] == 4 and r0[2] == 21) else 'FAIL'}")

    # ---------------- R3 threshold sweep ----------------------------------
    print("\n" + "=" * 72)
    print("R3 AREA-PREFILTER THRESHOLD SWEEP (then conf-sort top-5):")
    print(f"  {'thr':>5} | novel(6 dense) | base | stuff_slots")
    sweep = {}
    for thr in (0.10, 0.15, 0.20, 0.25):
        res = run_rule(lambda d: d["conf"], prefilter=lambda d, t=thr: d["barea"] / d["img_area"] <= t)
        nov, bas, stuff = totals(res)
        sweep[thr] = (nov, bas, stuff, res)
        print(f"  {thr:>5.2f} |       {nov:>2}       |  {bas:>2}  |     {stuff}")

    # ---------------- summary table ---------------------------------------
    print("\n" + "#" * 72)
    print("SUMMARY")
    print(f"  {'Rule':28s} | novel_rec(6 dense) | stuff_slots/35 | base_rec")
    summ = {}
    for name, res in rules.items():
        nov, bas, stuff = totals(res)
        summ[name] = (nov, bas, stuff)
        print(f"  {name:28s} |        {nov:>2}          |      {stuff:>2}        |   {bas:>2}")

    # ---------------- best rule breakdown ---------------------------------
    # best = max novel recovered, tie-break min stuff
    best_name = max(summ, key=lambda k: (summ[k][0], -summ[k][2]))
    bnov, bbas, bstuff = summ[best_name]
    res = rules[best_name]
    base_novel, base_stuff = 4, 21
    print("\n" + "#" * 72)
    print(f"BEST RULE: {best_name}")
    print(f"  novel recovered in top-5 = {bnov}  (baseline 4)  -> +{bnov - base_novel}")
    print(f"  stuff slots = {bstuff}/35  (baseline 21)  -> {bstuff - base_stuff:+d}")
    print(f"  of the 67 detected-but-cut novel objects, this rule recovers "
          f"{bnov - base_novel} more into top-5")
    print(f"  {'img':>7} | novel_found(diag) | novel_rec(base->best) | stuff(base->best)")
    diag_found = {579329: 12, 122263: 11, 157105: 1, 443084: 12, 92869: 21, 475808: 14, 34: 0}
    r0res = rules["R0 baseline (conf)"]
    for iid in IMAGES:
        nb = r0res[iid][0]; nr = res[iid][0]
        sb = r0res[iid][2]; sr = res[iid][2]
        flag = ""
        if iid == 157105:
            flag = "  <- recall-limited: only 1/22 novel detected; ceiling=1"
        if iid == 34:
            flag = "  <- no survivor matches zebra; ceiling=0"
        print(f"  {iid:>7} |        {diag_found[iid]:>2}         |     {nb} -> {nr}            |   {sb} -> {sr}{flag}")

    # ---------------- honesty checks --------------------------------------
    print("\n" + "#" * 72)
    print("HONESTY CHECKS (no rule can beat detection):")
    for iid, ceil in ((157105, 1), (34, 0)):
        vals = {n: r[iid][0] for n, r in rules.items()}
        bad = {n: v for n, v in vals.items() if v > ceil}
        print(f"  {iid}: novel_rec per rule = {vals}  ceiling={ceil}  "
              f"{'OK (no false gains)' if not bad else 'VIOLATION: ' + str(bad)}")
    print("\nSTOP — simulation only. No training, no precompute changes.")


if __name__ == "__main__":
    main()
