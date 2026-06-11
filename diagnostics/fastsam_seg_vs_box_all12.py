"""
FastSAM SEGMENT-EVERYTHING vs BOXES — all 12 diagnostic images. Visualization +
counts + loss-mechanism analysis. Inspection only, no logic changes.

Same task/settings as the 5-image run (FastSAM-x, conf=0.10, iou=0.7, imgsz=1024,
retina_masks=True; everything-mode masks, NO filter/cap on the mask panels). The
filter + cap are applied only to derive M survivors and the 5 kept.

Faithfulness:
  * survivor count M is HARD-ASSERTED against the ranking diagnostic for the 7
    images it covered; abort on mismatch.
  * the top-5 kept is HARD-ASSERTED against the saved _filtered.json for EVERY
    image present in it (the actual pipeline output); abort on mismatch.
  * the 5 new images (no diagnostic M reference) are reported; their kept top-5
    still gets the saved-file assert.

Loss-mechanism per image traces each novel GT's fate against everything-boxes /
survivors / kept, with the geometric-filter drop reason, to classify the dominant
loss as (i) K=5 cap, (ii) area filter, or (iii) recall.

STOP after the 12 figures + per-image table + aggregate summary.
"""

import gc
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import cv2

from fastsam_rank_diag import (
    DEVICE, WEIGHTS, CONF, IOU, IMGSZ,
    MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT, IOU_MATCH,
    iou_1ton, gpu_guard, load_base_names, tier_of, FULL_GT, FILT,
)

ALL12 = [157105, 122263, 543882, 579329, 443084, 92869, 475808,
         435091, 171270, 307238, 30, 34]
# survivor (pre-cap) counts from the ranking diagnostic (only 7 of the 12)
EXPECT = {579329: 291, 122263: 292, 157105: 298, 443084: 212,
          92869: 139, 475808: 182, 34: 12}
TOPK = 5
OUTDIR = Path("diagnostics/output/fastsam_seg_vs_box")
IMG_DIR = Path("data/Images/train2017")

TIER_COLOR = {"novel": (0.13, 0.85, 0.13), "base": (1.00, 0.60, 0.00),
              "other": (0.60, 0.60, 0.60), "stuff": (0.95, 0.15, 0.15)}


def load_gt_12(base):
    print(f"Loading full GT {FULL_GT} (one-time) ...")
    import time
    t = time.time()
    d = json.load(open(FULL_GT))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    want = set(ALL12)
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
    print(f"  GT loaded in {time.time()-t:.0f}s  (images found: {sorted(wh)})")
    return gt, wh


def fastsam_everything(model, img_path, iw, ih):
    """Raw 'everything' inference. Returns N total masks, the bool mask stack, the
    survivors (post-filter, conf-desc), and ALL detections with their drop reason."""
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, verbose=False)
    r = res[0]
    boxes, masks = r.boxes, r.masks
    assert masks is not None and len(masks.data) == len(boxes)
    mdata = (masks.data > 0.5).detach().cpu().numpy()
    n_total = len(mdata)
    image_area = iw * ih
    all_dets = []  # (x1,y1,x2,y2,conf,reason)  reason None == survivor
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy[0].tolist()
        c = float(boxes[i].conf[0])
        bw, bh = x2 - x1, y2 - y1
        reason = None
        if bw < MIN_SIDE or bh < MIN_SIDE:
            reason = "min-side"
        elif bw * bh > MAX_AREA_RATIO * image_area:
            reason = "area"
        elif max(bw / bh, bh / bw) > MAX_ASPECT:
            reason = "aspect"
        all_dets.append((x1, y1, x2, y2, c, reason))
    surv = [d for d in all_dets if d[5] is None]
    surv.sort(key=lambda d: d[4], reverse=True)
    return n_total, mdata, surv, all_dets


def seg_overlay(img, mdata, iw, ih, alpha=0.6, seed=0):
    rng = np.random.default_rng(seed)
    N = len(mdata)
    colors = rng.uniform(0.25, 1.0, size=(max(N, 1), 3))
    order = np.argsort([-int(m.sum()) for m in mdata]) if N else []
    seg = np.zeros((ih, iw, 3), dtype=np.float64)
    has = np.zeros((ih, iw), dtype=bool)
    for i in order:
        m = mdata[i]
        if m.shape != (ih, iw):
            m = cv2.resize(m.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
        seg[m] = colors[i]
        has[m] = True
    out = img.astype(np.float64).copy()
    out[has] = (1 - alpha) * out[has] + alpha * seg[has] * 255.0
    return out.astype(np.uint8), int(has.sum())


def tier_of_box(box, G, g_tier, g_name):
    if len(G) == 0:
        return "stuff", "-", 0.0
    iou = iou_1ton(np.array(box, dtype=np.float64), G)
    j = int(np.argmax(iou))
    if iou[j] >= IOU_MATCH:
        return g_tier[j], g_name[j], float(iou[j])
    return "stuff", "-", float(iou[j])


def fate(gt_list, Ball, reasons_all, Bsurv, Bkept, want_tier):
    """Trace each GT (of tier want_tier, or 'any') to its best fate:
    kept > cap(survivor, not kept) > filter(detected, dropped) > recall(never)."""
    res = dict(total=0, kept=0, cap=0, filter_area=0, filter_other=0, recall=0, lost=[])
    for gb, name, t in gt_list:
        if want_tier != "any" and t != want_tier:
            continue
        res["total"] += 1
        if len(Bkept) and float(iou_1ton(gb, Bkept).max()) >= IOU_MATCH:
            res["kept"] += 1
            continue
        if len(Bsurv) and float(iou_1ton(gb, Bsurv).max()) >= IOU_MATCH:
            res["cap"] += 1; res["lost"].append((name, "cap"))
            continue
        if len(Ball):
            ious = iou_1ton(gb, Ball)
            j = int(ious.argmax())
            if float(ious[j]) >= IOU_MATCH:
                rj = reasons_all[j] or "??"
                if rj == "area":
                    res["filter_area"] += 1; res["lost"].append((name, "filter:area"))
                else:
                    res["filter_other"] += 1; res["lost"].append((name, f"filter:{rj}"))
                continue
        res["recall"] += 1; res["lost"].append((name, "recall"))
    return res


def classify(novel_f, any_f):
    """dominant loss mechanism + basis."""
    basis = "novel" if novel_f["total"] > 0 else "all-GT"
    f = novel_f if basis == "novel" else any_f
    if f["total"] == 0:
        return "n/a (no GT objects)", basis, (0, 0, 0)
    cap = f["cap"]
    area = f["filter_area"]
    rec = f["recall"] + f["filter_other"]
    if cap + area + rec == 0:
        return "none (all kept)", basis, (cap, area, rec)
    label = max((("(i) K=5 cap", cap), ("(ii) area filter", area), ("(iii) recall", rec)),
                key=lambda x: x[1])[0]
    return label, basis, (cap, area, rec)


def draw_figure(iid, img, n_total, coverage, surv, kept_info, iw, ih, seg_img):
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.6))
    fig.suptitle(f"IMAGE {iid}  ({iw}x{ih})   funnel:  {n_total} masks  ->  "
                 f"{len(surv)} survivor boxes  ->  5 kept", fontsize=14, y=1.02)
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(seg_img); axes[1].set_title(f"FastSAM: {n_total} total masks"); axes[1].axis("off")
    axes[1].text(0.5, -0.04, f"masks cover {100*coverage:.1f}% of pixels",
                 transform=axes[1].transAxes, ha="center", fontsize=10)
    axes[2].imshow(img)
    for d in surv:
        x1, y1, x2, y2 = d[0], d[1], d[2], d[3]
        axes[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=(0, 1, 1), lw=1.0))
    axes[2].set_title(f"{len(surv)} survivor boxes"); axes[2].axis("off")
    axes[3].imshow(img)
    for box, tier, name, miou in kept_info:
        x1, y1, x2, y2 = box
        col = TIER_COLOR[tier]
        axes[3].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=col, lw=2.6))
        lab = f"{name}({tier})" if name != "-" else "stuff/bg"
        axes[3].text(x1, max(0, y1 - 3), lab, color="white", fontsize=8,
                     bbox=dict(fc=col, ec="none", alpha=0.85, pad=0.6))
    axes[3].set_title(f"5 kept (of {n_total} masks)"); axes[3].axis("off")
    axes[3].legend(handles=[Patch(color=TIER_COLOR[t], label=t)
                            for t in ("novel", "base", "other", "stuff")],
                   loc="lower right", fontsize=8, framealpha=0.85)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{iid}_segbox.png"
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    gpu_guard()
    base = load_base_names()
    gt, wh = load_gt_12(base)

    saved_fb = defaultdict(set)
    want = set(ALL12)
    for a in json.load(open(FILT)):
        if a["image_id"] in want:
            saved_fb[a["image_id"]].add(tuple(round(v, 4) for v in a["bbox"]))

    # upfront existence check (fail fast before model load)
    missing = [iid for iid in ALL12 if not (IMG_DIR / f"{iid:012d}.jpg").exists()]
    if missing:
        raise SystemExit(f"missing images: {missing}")
    no_gt = [iid for iid in ALL12 if iid not in wh]
    if no_gt:
        raise SystemExit(f"images not in GT json: {no_gt}")

    from ultralytics import FastSAM
    model = FastSAM(WEIGHTS)
    print(f"FastSAM loaded from {WEIGHTS}\n")

    rows = []
    for iid in ALL12:
        iw, ih = wh[iid]
        img_path = str(IMG_DIR / f"{iid:012d}.jpg")
        n_total, mdata, surv, all_dets = fastsam_everything(model, img_path, iw, ih)
        M = len(surv)

        # ---- faithfulness ----------------------------------------------------
        if iid in EXPECT and M != EXPECT[iid]:
            raise SystemExit(f"ABORT {iid}: survivors={M} != diagnostic {EXPECT[iid]}")
        kept = surv[:TOPK]
        mine_top5 = {tuple(round(v, 4) for v in (d[0], d[1], d[2] - d[0], d[3] - d[1])) for d in kept}
        if iid in saved_fb:
            if mine_top5 != saved_fb[iid]:
                raise SystemExit(f"ABORT {iid}: recomputed top-5 != saved _filtered.json")
            kept_ok = "OK"
        else:
            kept_ok = "no-saved-ref"
        if n_total < M:
            raise SystemExit(f"ABORT {iid}: N={n_total} < M={M} (impossible)")

        # ---- GT-derived structures ------------------------------------------
        G = np.array([g[0] for g in gt[iid]], dtype=np.float64) if gt[iid] else np.zeros((0, 4))
        g_name = [g[1] for g in gt[iid]]
        g_tier = [g[2] for g in gt[iid]]
        kept_info = [(d[:4], *tier_of_box(d[:4], G, g_tier, g_name)) for d in kept]
        tc = Counter(t for _, t, _, _ in kept_info)

        # ---- loss-mechanism fate --------------------------------------------
        Ball = np.array([[d[0], d[1], d[2], d[3]] for d in all_dets], dtype=np.float64)
        reasons_all = [d[5] for d in all_dets]
        Bsurv = np.array([[d[0], d[1], d[2], d[3]] for d in surv], dtype=np.float64) if surv else np.zeros((0, 4))
        Bkept = np.array([list(d[:4]) for d in kept], dtype=np.float64) if kept else np.zeros((0, 4))
        nov_f = fate(gt[iid], Ball, reasons_all, Bsurv, Bkept, "novel")
        any_f = fate(gt[iid], Ball, reasons_all, Bsurv, Bkept, "any")
        mech, basis, _ = classify(nov_f, any_f)

        # ---- figure ----------------------------------------------------------
        img = cv2.imread(img_path)[:, :, ::-1]
        seg_img, covered = seg_overlay(img, mdata, iw, ih)
        coverage = covered / float(iw * ih)
        draw_figure(iid, img, n_total, coverage, surv, kept_info, iw, ih, seg_img)

        rows.append(dict(iid=iid, iw=iw, ih=ih, N=n_total, M=M, cov=coverage,
                         tc=tc, kept_info=kept_info, nov=nov_f, anyf=any_f,
                         mech=mech, basis=basis, kept_ok=kept_ok))
        del mdata, seg_img, img
        gc.collect()

    # ============================ per-image table ============================
    print("\n" + "#" * 92)
    print("PER-IMAGE — funnel + coverage + kept land-on split")
    print(f"  {'image':>7} | {'N':>4} | {'cov%':>5} | {'M':>4} | {'filt':>4} | {'cap':>4} | "
          f"{'kept stuff/base/novel/other':>27} | {'top5':>5}")
    for r in rows:
        s = r["tc"]
        split = f"{s.get('stuff',0)}/{s.get('base',0)}/{s.get('novel',0)}/{s.get('other',0)}"
        mref = "" if r["iid"] in EXPECT else "  (no diag-M ref)"
        print(f"  {r['iid']:>7} | {r['N']:>4} | {100*r['cov']:>4.1f} | {r['M']:>4} | "
              f"{r['N']-r['M']:>4} | {r['M']-5:>4} | {split:>27} | {r['kept_ok']:>5}{mref}")

    # ===================== per-image novel-fate table ========================
    print("\n" + "#" * 92)
    print("PER-IMAGE — novel-object fate (why novel objects don't reach the 5 kept):")
    print(f"  {'image':>7} | {'novelGT':>7} | {'kept':>4} | {'cap':>4} | {'filt-area':>9} | "
          f"{'filt-oth':>8} | {'recall':>6} | dominant loss (basis)")
    for r in rows:
        f = r["nov"]
        print(f"  {r['iid']:>7} | {f['total']:>7} | {f['kept']:>4} | {f['cap']:>4} | "
              f"{f['filter_area']:>9} | {f['filter_other']:>8} | {f['recall']:>6} | "
              f"{r['mech']} [{r['basis']}]")

    # ============================ aggregate ==================================
    covs = np.array([r["cov"] for r in rows])
    sumN = sum(r["N"] for r in rows)
    sumM = sum(r["M"] for r in rows)
    slot = Counter()
    for r in rows:
        slot.update(r["tc"])
    total_slots = 5 * len(rows)
    novel_kept_imgs = []
    novel_kept_total = 0
    for r in rows:
        nk = [name for _, t, name, _ in r["kept_info"] if t == "novel"]
        if nk:
            novel_kept_total += len(nk)
            novel_kept_imgs.append((r["iid"], Counter(nk)))

    buckets = defaultdict(list)
    for r in rows:
        buckets[r["mech"]].append(r["iid"])

    print("\n" + "#" * 92)
    print("AGGREGATE SUMMARY (12 images)")
    print(f"  pixel coverage: mean={100*covs.mean():.1f}%  min={100*covs.min():.1f}%  "
          f"max={100*covs.max():.1f}%   -> FastSAM segments essentially the whole image")
    print(f"  FUNNEL (summed): {sumN} total masks -> {sumM} survivor boxes -> {total_slots} kept "
          f"(={total_slots/sumN*100:.1f}% of masks survive to a label)")
    print(f"  filter drops total = {sumN - sumM}   cap drops total = {sumM - total_slots}")
    print(f"  KEPT-SLOT breakdown across {total_slots} slots: "
          f"stuff={slot.get('stuff',0)} base={slot.get('base',0)} "
          f"novel={slot.get('novel',0)} other={slot.get('other',0)}")
    print(f"  NOVEL objects kept total = {novel_kept_total}, from images:")
    for iid, c in novel_kept_imgs:
        print(f"      {iid}: {dict(c)}")
    print("\n  LOSS-MECHANISM TALLY (dominant per image):")
    for mech in ("(i) K=5 cap", "(ii) area filter", "(iii) recall",
                 "none (all kept)", "n/a (no GT objects)"):
        if buckets.get(mech):
            print(f"    {mech:22s}: {len(buckets[mech])} imgs -> {sorted(buckets[mech])}")
    other = {m: v for m, v in buckets.items() if m not in
             ("(i) K=5 cap", "(ii) area filter", "(iii) recall", "none (all kept)", "n/a (no GT objects)")}
    for m, v in other.items():
        print(f"    {m:22s}: {len(v)} imgs -> {sorted(v)}")

    # ===================== confirm all 12 figures exist ======================
    present = sorted(int(p.stem.split("_")[0]) for p in OUTDIR.glob("*_segbox.png"))
    have_all = set(ALL12).issubset(set(present))
    print("\n" + "#" * 92)
    print(f"FIGURES: {len([i for i in ALL12 if i in present])}/12 of the requested IDs present in {OUTDIR}/")
    print(f"  all 12 exist: {'YES' if have_all else 'NO -> missing ' + str(sorted(set(ALL12)-set(present)))}")
    print("STOP — visualization only, no filtering changes, no training.")


if __name__ == "__main__":
    main()
