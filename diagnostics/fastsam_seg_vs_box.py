"""
FastSAM WHOLE-IMAGE SEGMENTATION vs BOXES — visualization, inspection only.

Shows FastSAM's full "segment everything" output (ALL masks) SEPARATELY from the
boxes the pseudo-label pipeline keeps, so the funnel is visible:
    original  ->  everything segmented (N masks)  ->  all survivor boxes (M)  ->  5 kept

Settings identical to precompute: FastSAM-x, conf=0.10, iou=0.7, imgsz=1024,
retina_masks=True. For the mask panels NO geometric filter and NO top-5 cap are
applied — that is the whole point (N = everything). The filter + cap are applied
only to derive M survivors and the 5 kept, whose counts are HARD-ASSERTED against
the ranking diagnostic (abort on mismatch).

STOP after the figures + counts. No filtering changes, no training.
"""

import gc
import json
from collections import defaultdict
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
    iou_1ton, gpu_guard, load_base_names, load_gt, FILT,
)

SEG_IMAGES = [579329, 122263, 92869, 475808, 34]
EXPECT = {579329: 291, 122263: 292, 92869: 139, 475808: 182, 34: 12}  # survivors (diagnostic)
TOPK = 5
OUTDIR = Path("diagnostics/output/fastsam_seg_vs_box")
IMG_DIR = Path("data/Images/train2017")

TIER_COLOR = {"novel": (0.13, 0.85, 0.13), "base": (1.00, 0.60, 0.00),
              "other": (0.60, 0.60, 0.60), "stuff": (0.95, 0.15, 0.15)}


def fastsam_everything(model, img_path, iw, ih):
    """Raw 'everything' inference. Returns (N total masks, bool mask stack,
    survivors[post-filter, conf-desc] with mask index)."""
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, verbose=False)
    r = res[0]
    boxes = r.boxes
    masks = r.masks
    assert masks is not None and len(masks.data) == len(boxes)
    mdata = (masks.data > 0.5).detach().cpu().numpy()  # (N,mh,mw) bool — ALL masks
    n_total = len(mdata)
    image_area = iw * ih
    surv = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].xyxy[0].tolist()
        c = float(boxes[i].conf[0])
        bw, bh = x2 - x1, y2 - y1
        if bw < MIN_SIDE or bh < MIN_SIDE:
            continue
        if bw * bh > MAX_AREA_RATIO * image_area:
            continue
        if max(bw / bh, bh / bw) > MAX_ASPECT:
            continue
        surv.append(dict(box=(x1, y1, x2, y2), conf=c, mi=i))
    surv.sort(key=lambda d: d["conf"], reverse=True)
    return n_total, mdata, surv


def seg_overlay(img, mdata, iw, ih, alpha=0.6, seed=0):
    """Paint every mask a distinct random color, big->small so small ones stay
    visible. Returns (overlaid uint8 image, covered-pixel count)."""
    rng = np.random.default_rng(seed)
    N = len(mdata)
    colors = rng.uniform(0.25, 1.0, size=(N, 3))
    order = np.argsort([-int(m.sum()) for m in mdata])  # large area first
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


def draw_figure(iid, img, n_total, coverage, surv, kept_info, iw, ih, seg_img):
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.6))
    fig.suptitle(f"IMAGE {iid}  ({iw}x{ih})   funnel:  {n_total} masks  ->  "
                 f"{len(surv)} survivor boxes  ->  5 kept", fontsize=14, y=1.02)

    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")

    axes[1].imshow(seg_img)
    axes[1].set_title(f"FastSAM: {n_total} total masks")
    axes[1].axis("off")
    axes[1].text(0.5, -0.04, f"masks cover {100*coverage:.1f}% of pixels",
                 transform=axes[1].transAxes, ha="center", fontsize=10)

    axes[2].imshow(img)
    for s in surv:
        x1, y1, x2, y2 = s["box"]
        axes[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=(0, 1, 1), lw=1.0))
    axes[2].set_title(f"{len(surv)} survivor boxes")
    axes[2].axis("off")

    axes[3].imshow(img)
    for box, tier, name, miou in kept_info:
        x1, y1, x2, y2 = box
        col = TIER_COLOR[tier]
        axes[3].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=col, lw=2.6))
        lab = f"{name}({tier})" if name != "-" else "stuff/bg"
        axes[3].text(x1, max(0, y1 - 3), lab, color="white", fontsize=8,
                     bbox=dict(fc=col, ec="none", alpha=0.85, pad=0.6))
    axes[3].set_title(f"5 kept (of {n_total} masks)")
    axes[3].axis("off")
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
    gt, wh = load_gt(base)

    saved_fb = defaultdict(set)
    want = set(SEG_IMAGES)
    for a in json.load(open(FILT)):
        if a["image_id"] in want:
            saved_fb[a["image_id"]].add(tuple(round(v, 4) for v in a["bbox"]))

    from ultralytics import FastSAM
    model = FastSAM(WEIGHTS)
    print(f"FastSAM loaded from {WEIGHTS}\n")

    rows = []  # (iid, N, M, kept, coverage, tiers, fig)
    for iid in SEG_IMAGES:
        iw, ih = wh[iid]
        img_path = str(IMG_DIR / f"{iid:012d}.jpg")
        if not Path(img_path).exists():
            raise SystemExit(f"missing image {img_path}")

        n_total, mdata, surv = fastsam_everything(model, img_path, iw, ih)
        # ---- faithfulness: survivors + kept must match the ranking diagnostic --
        if len(surv) != EXPECT[iid]:
            raise SystemExit(f"ABORT {iid}: survivors={len(surv)} != diagnostic {EXPECT[iid]}")
        kept = surv[:TOPK]
        mine_top5 = {tuple(round(v, 4) for v in (s["box"][0], s["box"][1],
                     s["box"][2] - s["box"][0], s["box"][3] - s["box"][1])) for s in kept}
        if mine_top5 != saved_fb[iid]:
            raise SystemExit(f"ABORT {iid}: recomputed top-5 != saved _filtered.json")
        if n_total < len(surv):
            raise SystemExit(f"ABORT {iid}: N total {n_total} < survivors {len(surv)} (impossible)")

        G = np.array([g[0] for g in gt[iid]], dtype=np.float64) if gt[iid] else np.zeros((0, 4))
        g_name = [g[1] for g in gt[iid]]
        g_tier = [g[2] for g in gt[iid]]
        kept_info = []
        for s in kept:
            tier, name, miou = tier_of_box(s["box"], G, g_tier, g_name)
            kept_info.append((s["box"], tier, name, miou))

        img = cv2.imread(img_path)[:, :, ::-1]
        seg_img, covered = seg_overlay(img, mdata, iw, ih)
        coverage = covered / float(iw * ih)
        fig = draw_figure(iid, img, n_total, coverage, surv, kept_info, iw, ih, seg_img)

        tiers = [t for _, t, _, _ in kept_info]
        from collections import Counter
        tc = Counter(tiers)
        print("=" * 78)
        print(f"IMAGE {iid}  ({iw}x{ih})")
        print(f"  total masks (everything mode)      N = {n_total}")
        print(f"  survivor boxes (post geom-filter)  M = {len(surv)}   (diagnostic {EXPECT[iid]} -> match)")
        print(f"  kept by pseudo-label (top-5 conf)    = {len(kept)}   (top-5 == saved _filtered.json -> match)")
        print(f"  funnel: {n_total} -> {len(surv)} -> 5   "
              f"(filter drops {n_total - len(surv)} masks, cap drops {len(surv) - 5})")
        print(f"  everything-masks cover {100*coverage:.1f}% of image pixels")
        print(f"  5 kept land on: stuff={tc.get('stuff',0)} base={tc.get('base',0)} "
              f"novel={tc.get('novel',0)} other={tc.get('other',0)}")
        for box, tier, name, miou in kept_info:
            tag = f"{name}({tier}) iou={miou:.2f}" if name != "-" else f"stuff/bg (best iou={miou:.2f})"
            print(f"      kept -> {tag}")
        print(f"  figure -> {fig}")
        rows.append((iid, n_total, len(surv), len(kept), coverage, tc, str(fig)))

        del mdata, seg_img, img
        gc.collect()

    # ---------------- counts table ----------------------------------------
    print("\n" + "#" * 78)
    print("COUNTS — the funnel per image (everything -> survivors -> kept):")
    print(f"  {'image':>7} | {'N total masks':>13} | {'M survivors':>11} | {'kept':>4} | "
          f"{'coverage':>8} | match")
    for iid, N, M, k, cov, tc, fig in rows:
        ok = (M == EXPECT[iid] and k == 5)
        print(f"  {iid:>7} | {N:>13} | {M:>11} | {k:>4} | {100*cov:>7.1f}% | "
              f"{'OK' if ok else 'MISMATCH'}")
    print("\n  (M survivors + kept=5 asserted == ranking diagnostic for every image; "
          "would have aborted on mismatch)")
    print("STOP — visualization only, no filtering changes, no training.")


if __name__ == "__main__":
    main()
