"""
FastSAM max_det SENSITIVITY CHECK — inspection only, no logic/training changes.

The 12-image run flagged 3 images (157105, 122263, 171270) hitting the ultralytics
default max_det=300, truncating their 'everything' output before selection. Their
novel loss is attributed to "recall" (no detection box at IoU>=0.5). Question: is
some of that "recall" actually max_det=300 truncation?

Re-run everything mode on ONLY these 3 with max_det=1000 (all else identical:
FastSAM-x, conf=0.10, iou=0.7, imgsz=1024, retina_masks=True). max_det is applied
AFTER NMS as a top-k-by-conf truncation, so B300 is a strict conf-ordered prefix of
B1000 -> the extra detections are all lower-conf than the kept top-5, and any novel
GT "recovered" at 1000 is matched specifically by a rank>300 (truncated) box.

Faithfulness at max_det=300: N==300, survivors==diagnostic, top-5==saved, and the
recall-bucket count reproduces the 12-image run. Abort on mismatch.

STOP after the 3-image comparison + interpretation. No precompute/training changes.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fastsam_rank_diag import (
    DEVICE, WEIGHTS, CONF, IOU, IMGSZ,
    MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT, IOU_MATCH,
    iou_1ton, gpu_guard, load_base_names, tier_of, FULL_GT, FILT,
)

IMAGES = [157105, 122263, 171270]
EXPECT_SURV = {157105: 298, 122263: 292, 171270: 299}     # survivors @300 (diagnostic)
KNOWN_RECALL300 = {157105: 21, 122263: 15, 171270: 3}      # recall-bucket @300 (12-img run)
KNOWN_NOVEL = {157105: 22, 122263: 26, 171270: 3}          # novel GT per image
IMG_DIR = Path("data/Images/train2017")
TOPK = 5


def load_gt_novel(base):
    d = json.load(open(FULL_GT))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    want = set(IMAGES)
    wh = {im["id"]: (im["width"], im["height"]) for im in d["images"] if im["id"] in want}
    gt = defaultdict(list)
    for a in d["annotations"]:
        iid = a["image_id"]
        if iid not in want:
            continue
        x, y, bw, bh = a["bbox"]
        name = cats[a["category_id"]]
        gt[iid].append((np.array([x, y, x + bw, y + bh], dtype=np.float64),
                        name, tier_of(name, base)))
    return gt, wh


def everything_boxes(model, img_path, iw, ih, max_det):
    """Return conf-desc native boxes (M,4), their confs (M,), and post-geom-filter
    survivors (sorted conf-desc)."""
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, max_det=max_det, verbose=False)
    r = res[0]
    boxes = r.boxes
    B, confs, surv = [], [], []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy[0].tolist()
        c = float(boxes[i].conf[0])
        B.append([x1, y1, x2, y2]); confs.append(c)
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if bw * bh > MAX_AREA_RATIO * iw * ih:
            continue
        if max(bw / bh, bh / bw) > MAX_ASPECT:
            continue
        surv.append((x1, y1, x2, y2, c))
    surv.sort(key=lambda t: t[4], reverse=True)
    return np.array(B, dtype=np.float64), np.array(confs, dtype=np.float64), surv


def main():
    gpu_guard()
    base = load_base_names()
    gt, wh = load_gt_novel(base)

    saved_fb = defaultdict(set)
    for a in json.load(open(FILT)):
        if a["image_id"] in set(IMAGES):
            saved_fb[a["image_id"]].add(tuple(round(v, 4) for v in a["bbox"]))

    from ultralytics import FastSAM
    model = FastSAM(WEIGHTS)
    print(f"FastSAM loaded from {WEIGHTS}\n")

    tot_recall300 = tot_recall1000 = tot_recovered = tot_trunc = 0
    rows = []
    for iid in IMAGES:
        iw, ih = wh[iid]
        img_path = str(IMG_DIR / f"{iid:012d}.jpg")

        B300, c300, surv300 = everything_boxes(model, img_path, iw, ih, 300)
        B1000, c1000, surv1000 = everything_boxes(model, img_path, iw, ih, 1000)

        # ---- faithfulness of the 300 baseline ------------------------------
        assert len(B300) == 300, f"{iid}: N@300={len(B300)} != 300"
        assert len(surv300) == EXPECT_SURV[iid], f"{iid}: surv@300={len(surv300)} != {EXPECT_SURV[iid]}"
        top5_300 = {tuple(round(v, 4) for v in (s[0], s[1], s[2] - s[0], s[3] - s[1])) for s in surv300[:TOPK]}
        assert top5_300 == saved_fb[iid], f"{iid}: top-5@300 != saved"
        # pipeline output invariance: top-5 unchanged when max_det=1000
        top5_1000 = {tuple(round(v, 4) for v in (s[0], s[1], s[2] - s[0], s[3] - s[1])) for s in surv1000[:TOPK]}
        top5_invariant = (top5_1000 == saved_fb[iid])

        N1000 = len(B1000)
        truncated = N1000 - 300

        # ---- novel-GT recall: @300 vs @1000 --------------------------------
        novel = [(gb, nm) for gb, nm, t in gt[iid] if t == "novel"]
        recall300 = recall1000 = recovered = 0
        recovered_detail = []
        for gb, nm in novel:
            m300 = len(B300) and float(iou_1ton(gb, B300).max()) >= IOU_MATCH
            i1000 = int(iou_1ton(gb, B1000).argmax()) if len(B1000) else -1
            best1000 = float(iou_1ton(gb, B1000)[i1000]) if i1000 >= 0 else 0.0
            m1000 = best1000 >= IOU_MATCH
            if not m300:
                recall300 += 1
                if m1000:
                    recovered += 1
                    recovered_detail.append((nm, i1000 + 1, c1000[i1000], best1000))
                else:
                    recall1000 += 1
        assert recall300 == KNOWN_RECALL300[iid], \
            f"{iid}: recall@300={recall300} != 12-img-run {KNOWN_RECALL300[iid]}"

        tot_recall300 += recall300
        tot_recall1000 += recall1000
        tot_recovered += recovered
        tot_trunc += truncated
        rows.append((iid, N1000, truncated, recall300, recall1000, recovered,
                     top5_invariant, recovered_detail, c300.min(), c1000.min()))

        print("=" * 78)
        print(f"IMAGE {iid}  ({iw}x{ih})   novel GT = {KNOWN_NOVEL[iid]}")
        print(f"  total masks: 300 (max_det=300, TRUNCATED)  ->  {N1000} (max_det=1000)"
              f"   => {truncated} detections were being hidden past rank 300")
        print(f"  conf range: @300 min={c300.min():.3f}  @1000 min={c1000.min():.3f}  "
              f"(the {truncated} extra are all conf<= the 300th)")
        print(f"  pipeline top-5 invariant to max_det: {'YES' if top5_invariant else 'NO'}")
        print(f"  RECALL-bucket novel:  @300 = {recall300}   @1000 = {recall1000}   "
              f"=> RECOVERED by raising max_det = {recovered}")
        if recovered_detail:
            print(f"  recovered (now matched at IoU>=0.5 by a truncated box):")
            for nm, rank, cf, io in recovered_detail:
                print(f"      {nm}: matching box rank={rank} (>300 => was truncated)  "
                      f"conf={cf:.3f}  IoU={io:.2f}")

    # ---------------- summary + interpretation ----------------------------
    print("\n" + "#" * 78)
    print("SUMMARY (3 images)")
    print(f"  {'image':>7} | {'N@1000':>6} | {'truncated':>9} | {'recall@300':>10} | "
          f"{'recall@1000':>11} | {'recovered':>9} | top5-inv")
    for iid, N1000, trunc, r3, r10, rec, inv, det, _, _ in rows:
        print(f"  {iid:>7} | {N1000:>6} | {trunc:>9} | {r3:>10} | {r10:>11} | {rec:>9} | "
              f"{'yes' if inv else 'NO':>7}")
    print(f"  {'TOTAL':>7} | {'':>6} | {tot_trunc:>9} | {tot_recall300:>10} | "
          f"{tot_recall1000:>11} | {tot_recovered:>9} |")

    frac = tot_recovered / tot_recall300 if tot_recall300 else 0.0
    print("\n" + "#" * 78)
    print("INTERPRETATION")
    print(f"  recall-bucket novel @300 = {tot_recall300};  recovered by max_det=1000 = "
          f"{tot_recovered}  ({100*frac:.0f}% of the recall bucket)")
    if tot_recovered == 0:
        print("  => recall barely shrinks: the 54% recall finding is REAL. FastSAM genuinely")
        print("     does not carve these novel instances; it is robust to the max_det cap.")
    elif frac < 0.20:
        print(f"  => recall shrinks only slightly ({100*frac:.0f}%): the recall finding is")
        print("     largely REAL; a small part of it was max_det=300 truncation.")
    else:
        print(f"  => recall shrinks substantially ({100*frac:.0f}%): a meaningful part of the")
        print("     'recall' loss was a max_det=300 truncation ARTIFACT. The true recall-vs-cap")
        print("     split needs revising (these recovered objects ARE detected, just past rank 300).")
    print("  NOTE: the pipeline's kept top-5 is unchanged by max_det (recovered boxes are all")
    print("        lower-conf than the kept 5) — this probe reclassifies the LOSS mechanism,")
    print("        it does not by itself recover any novel object into the pseudo-labels.")
    print("\nSTOP — diagnostic only, no precompute or training changes.")


if __name__ == "__main__":
    main()
