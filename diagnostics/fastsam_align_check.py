"""
FastSAM MASK-BOX ALIGNMENT CHECK — inspection only, no logic changes.

FastSAM emits detection boxes (box.xyxy, detection head) and masks (seg head) as
SEPARATE tensors. This verifies, per detection, that a box and ITS OWN mask agree
spatially. Not about GT — box-vs-its-own-mask.

ALL survivors are checked (NOT capped to 5, NOT re-ranked) so the alignment test
is not biased by selection. The saved _filtered.json holds only the post-cap top-5
and never stored masks, so FastSAM is re-run to recover the pre-cap survivors WITH
their masks. FastSAM is deterministic here (the ranking diagnostic proved the top-5
reproduce exactly), so the regenerated survivor counts are HARD-ASSERTED against the
diagnostic per image; abort on mismatch.

Per detection:
  1. mask-derived box = tight axis-aligned bbox of the mask's nonzero pixels.
  2. IoU(mask-derived box, native box.xyxy).
  3. centroid distance (mask centroid vs box center) / box diagonal.
  4. containment = fraction of mask pixels inside the native box.

STOP after the report + figures. Inspection only.
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

from fastsam_rank_diag import (
    DEVICE, WEIGHTS, CONF, IOU, IMGSZ,
    MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT, IOU_MATCH,
    iou_1ton, gpu_guard, load_base_names, load_gt, FILT,
)

ALIGN_IMAGES = [579329, 122263, 92869, 475808, 34]
# survivor (pre-cap) counts the ranking diagnostic reported (faithfulness contract)
EXPECT = {579329: 291, 122263: 292, 92869: 139, 475808: 182, 34: 12}
TOPK = 5
OUTDIR = Path("diagnostics/output/fastsam_align")
IMG_DIR = Path("data/Images/train2017")

# alignment thresholds for the count metrics
IOU_HI = 0.90          # "well aligned"
IOU_LO = 0.50          # "misaligned"
CONTAIN_MIN = 0.80     # mask spilling out of its box


def survivors_with_masks(model, img_path, iw, ih):
    """Re-run FastSAM, apply the geometric filter, return survivors (sorted by
    conf desc) each carrying the index of its mask, plus the bool mask stack."""
    res = model(img_path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
                retina_masks=True, verbose=False)
    r = res[0]
    boxes = r.boxes
    masks = r.masks
    has_mask = (masks is not None and getattr(masks, "data", None) is not None
                and len(masks.data) == len(boxes))
    mdata = (masks.data > 0.5).detach().cpu().numpy() if has_mask else None  # (N,mh,mw) bool
    image_area = iw * ih
    surv = []
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
        surv.append(dict(box=(x1, y1, x2, y2), conf=c, mi=i))
    surv.sort(key=lambda d: d["conf"], reverse=True)
    return surv, mdata, has_mask


def align_metrics(mask, box, iw, ih):
    """box-vs-its-own-mask metrics. mask is (mh,mw) bool; box xyxy in orig coords."""
    mh, mw = mask.shape
    sx, sy = iw / mw, ih / mh
    ys, xs = np.nonzero(mask)
    nb = np.array(box, dtype=np.float64)
    bcx, bcy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
    diag = float(np.hypot(nb[2] - nb[0], nb[3] - nb[1]))
    if len(xs) == 0:
        return dict(empty=True, npix=0, iou=0.0, cdist=float("nan"),
                    centroid_in_box=False, containment=0.0,
                    mbox=nb.copy(), centroid=(bcx, bcy))
    # mask-derived box -> original coords (+1 on far edge for pixel extent)
    mbox = np.array([xs.min() * sx, ys.min() * sy,
                     (xs.max() + 1) * sx, (ys.max() + 1) * sy], dtype=np.float64)
    iou = float(iou_1ton(nb, mbox[None, :])[0])
    cx, cy = xs.mean() * sx, ys.mean() * sy
    cdist = float(np.hypot(cx - bcx, cy - bcy) / diag) if diag > 0 else 0.0
    centroid_in_box = bool(nb[0] <= cx <= nb[2] and nb[1] <= cy <= nb[3])
    # containment in mask coords
    bx1, by1, bx2, by2 = nb[0] / sx, nb[1] / sy, nb[2] / sx, nb[3] / sy
    inside = (xs >= bx1) & (xs <= bx2) & (ys >= by1) & (ys <= by2)
    containment = float(inside.sum()) / len(xs)
    return dict(empty=False, npix=int(len(xs)), iou=iou, cdist=cdist,
                centroid_in_box=centroid_in_box, containment=containment,
                mbox=mbox, centroid=(cx, cy))


def landing(box, G, g_name, g_tier):
    """what a box lands on: 'name(tier)' if best GT IoU>=0.5 else 'stuff/bg'."""
    if len(G) == 0:
        return "stuff/bg", 0.0
    iou = iou_1ton(np.array(box, dtype=np.float64), G)
    j = int(np.argmax(iou))
    if iou[j] >= IOU_MATCH:
        return f"{g_name[j]}({g_tier[j]})", float(iou[j])
    return "stuff/bg", float(iou[j])


def full_mask(mask, iw, ih):
    if mask.shape == (ih, iw):
        return mask.astype(np.uint8)
    return cv2.resize(mask.astype(np.uint8), (iw, ih), interpolation=cv2.INTER_NEAREST)


def draw_overlay(ax, mask_full, color, alpha=0.4):
    rgba = np.zeros((*mask_full.shape, 4), dtype=np.float64)
    rgba[mask_full > 0] = (color[0], color[1], color[2], alpha)
    ax.imshow(rgba)


def draw_figure(iid, img, records, mdata, iw, ih):
    """Top panel: up to 12 detections (box solid + own mask, same color), sampled
    across the IoU range. Bottom: the worst-aligned detections zoomed, each with
    native box (solid) and mask-derived box (dashed) so divergence is visible."""
    order_iou = sorted(range(len(records)), key=lambda i: records[i]["iou"])
    # --- select up to 12 spanning the IoU range (best..worst), unbiased overview
    if len(records) <= 12:
        sel = list(range(len(records)))
    else:
        picks = sorted(set(np.linspace(0, len(records) - 1, 12).round().astype(int)))
        sel = [order_iou[k] for k in picks]
    # --- worst-aligned (lowest IoU, non-empty so a mask exists to show)
    worst = [i for i in order_iou if not records[i]["empty"]][:5]

    n_w = max(1, len(worst))
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, max(5, n_w), height_ratios=[3, 2])
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(img)
    ax_full.set_title(
        f"IMAGE {iid}  ({iw}x{ih})  —  {len(records)} survivors (ALL, pre-cap)  |  "
        f"showing {len(sel)} spanning the IoU range  |  box=solid, own mask=translucent same color",
        fontsize=11)
    ax_full.axis("off")
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, len(sel))))
    for c_i, ri in enumerate(sel):
        rec = records[ri]
        col = colors[c_i][:3]
        mf = full_mask(mdata[rec["mi"]], iw, ih)
        draw_overlay(ax_full, mf, col, alpha=0.40)
        x1, y1, x2, y2 = rec["box"]
        ax_full.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                    edgecolor=col, lw=2.0))
        ax_full.text(x1, max(0, y1 - 3), f"{rec['iou']:.2f}", color="white",
                     fontsize=8, bbox=dict(fc=col, ec="none", alpha=0.8, pad=0.5))

    for k in range(max(5, n_w)):
        ax = fig.add_subplot(gs[1, k])
        ax.axis("off")
        if k >= len(worst):
            continue
        rec = records[worst[k]]
        x1, y1, x2, y2 = rec["box"]
        mb = rec["mbox"]
        ux1, uy1 = min(x1, mb[0]), min(y1, mb[1])
        ux2, uy2 = max(x2, mb[2]), max(y2, mb[3])
        pad = 0.20 * max(ux2 - ux1, uy2 - uy1) + 5
        cx0, cy0 = int(max(0, ux1 - pad)), int(max(0, uy1 - pad))
        cx1, cy1 = int(min(iw, ux2 + pad)), int(min(ih, uy2 + pad))
        ax.imshow(img[cy0:cy1, cx0:cx1])
        mf = full_mask(mdata[rec["mi"]], iw, ih)[cy0:cy1, cx0:cx1]
        draw_overlay(ax, mf, (1.0, 0.2, 0.2), alpha=0.40)
        ax.add_patch(Rectangle((x1 - cx0, y1 - cy0), x2 - x1, y2 - y1, fill=False,
                               edgecolor="yellow", lw=2.0, label="native box"))
        ax.add_patch(Rectangle((mb[0] - cx0, mb[1] - cy0), mb[2] - mb[0], mb[3] - mb[1],
                               fill=False, edgecolor="cyan", lw=1.6, ls="--",
                               label="mask bbox"))
        ax.set_title(f"IoU={rec['iou']:.2f}  contain={rec['containment']*100:.0f}%\n"
                     f"{rec['land']}", fontsize=8)
    fig.text(0.5, 0.005,
             "bottom: worst-aligned, zoomed — yellow=native box(solid), "
             "cyan=mask-derived box(dashed), red=mask. divergence = misalignment.",
             ha="center", fontsize=9)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{iid}_align.png"
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out, len(sel), len(worst)


def main():
    gpu_guard()
    base = load_base_names()
    gt, wh = load_gt(base)

    saved_fb = defaultdict(set)
    want = set(ALIGN_IMAGES)
    for a in json.load(open(FILT)):
        if a["image_id"] in want:
            saved_fb[a["image_id"]].add(tuple(round(v, 4) for v in a["bbox"]))

    from ultralytics import FastSAM
    model = FastSAM(WEIGHTS)
    print(f"FastSAM loaded from {WEIGHTS}\n")

    manifest = []  # (iid, n_surv, expect, mh, mw, iw, ih, top5_ok, n_empty, fig, nshown, nworst)

    for iid in ALIGN_IMAGES:
        iw, ih = wh[iid]
        img_path = str(IMG_DIR / f"{iid:012d}.jpg")
        if not Path(img_path).exists():
            raise SystemExit(f"missing image {img_path}")

        surv, mdata, has_mask = survivors_with_masks(model, img_path, iw, ih)
        if not has_mask:
            raise SystemExit(f"ABORT {iid}: no per-detection masks returned — cannot do alignment check")
        # ---- faithfulness: ALL survivors must match the diagnostic count -------
        if len(surv) != EXPECT[iid]:
            raise SystemExit(f"ABORT {iid}: survivors={len(surv)} != diagnostic {EXPECT[iid]}")
        mine_top5 = {tuple(round(v, 4) for v in (s["box"][0], s["box"][1],
                     s["box"][2] - s["box"][0], s["box"][3] - s["box"][1]))
                     for s in surv[:TOPK]}
        top5_ok = (mine_top5 == saved_fb[iid])
        if not top5_ok:
            raise SystemExit(f"ABORT {iid}: recomputed top-5 != saved _filtered.json")
        mh, mw = mdata.shape[1], mdata.shape[2]

        G = np.array([g[0] for g in gt[iid]], dtype=np.float64) if gt[iid] else np.zeros((0, 4))
        g_name = [g[1] for g in gt[iid]]
        g_tier = [g[2] for g in gt[iid]]

        # ---- per-survivor alignment metrics (ALL survivors) -------------------
        records = []
        for s in surv:
            m = align_metrics(mdata[s["mi"]], s["box"], iw, ih)
            land_str, land_iou = landing(s["box"], G, g_name, g_tier)
            m.update(box=s["box"], conf=s["conf"], mi=s["mi"],
                     land=land_str, land_iou=land_iou)
            records.append(m)

        ious = np.array([r["iou"] for r in records if not r["empty"]])
        cont = np.array([r["containment"] for r in records if not r["empty"]])
        n_empty = sum(r["empty"] for r in records)
        n_eff = len(ious)
        centroid_out = sum((not r["centroid_in_box"]) for r in records)
        spill = int((cont < CONTAIN_MIN).sum()) + n_empty  # empty == 0% contained

        # ---------------- per-image report ------------------------------------
        print("=" * 80)
        print(f"IMAGE {iid}  ({iw}x{ih})   mask-res={mh}x{mw}"
              f"{'  (== image res)' if (mh, mw) == (ih, iw) else '  (scaled to image res for metrics)'}")
        print(f"  survivors checked (ALL, pre-cap) = {len(records)}   "
              f"empty-mask detections = {n_empty}")
        print(f"  MASK-vs-BOX IoU:  min={ious.min():.3f}  median={np.median(ious):.3f}  "
              f"mean={ious.mean():.3f}")
        print(f"    IoU >= {IOU_HI:.2f} (well aligned): {(ious>=IOU_HI).sum():>3}/{n_eff}  "
              f"= {100*(ious>=IOU_HI).mean():.1f}%")
        print(f"    IoU <  {IOU_LO:.2f} (misaligned)  : {(ious<IOU_LO).sum():>3}/{n_eff}  "
              f"= {100*(ious<IOU_LO).mean():.1f}%")
        print(f"  mask centroid OUTSIDE its own box (gross): {centroid_out}/{len(records)}")
        print(f"  mask >{int((1-CONTAIN_MIN)*100)}% OUTSIDE box (containment<{CONTAIN_MIN:.0%}): "
              f"{spill}/{len(records)}  (incl. {n_empty} empty)")
        print(f"  containment: min={cont.min()*100:.1f}%  median={np.median(cont)*100:.1f}%  "
              f"mean={cont.mean()*100:.1f}%")

        worst = sorted(records, key=lambda r: r["iou"])[:5]
        print("  WORST 5 (lowest mask-vs-box IoU):")
        print(f"    {'IoU':>5} {'contain':>7} {'cdist':>6} {'cen_in':>6}  conf   lands on")
        for r in worst:
            cd = "nan" if np.isnan(r["cdist"]) else f"{r['cdist']:.2f}"
            print(f"    {r['iou']:>5.2f} {r['containment']*100:>6.0f}% {cd:>6} "
                  f"{str(r['centroid_in_box']):>6}  {r['conf']:.3f}  {r['land']}")

        fig, nshown, nworst = draw_figure(iid, cv2.imread(img_path)[:, :, ::-1], records, mdata, iw, ih)
        print(f"  figure -> {fig}  (overview n={nshown}, worst-zoom n={nworst})")
        manifest.append((iid, len(records), EXPECT[iid], mh, mw, iw, ih,
                         top5_ok, n_empty, str(fig)))

        del mdata, records
        gc.collect()

    # ---------------- manifest --------------------------------------------
    print("\n" + "#" * 80)
    print("MANIFEST — regenerated survivor counts vs ranking diagnostic:")
    print(f"  {'image':>7} | {'survivors':>9} | {'diagnostic':>10} | match | "
          f"top5==saved | mask-res vs img | empty | figure")
    all_ok = True
    for iid, n, exp, mh, mw, iw, ih, t5, ne, fig in manifest:
        ok = (n == exp)
        all_ok &= ok and t5
        res_tag = "same" if (mh, mw) == (ih, iw) else f"{mh}x{mw}->{ih}x{iw}"
        print(f"  {iid:>7} | {n:>9} | {exp:>10} | {'OK' if ok else 'MISMATCH':>5} | "
              f"{'OK' if t5 else 'FAIL':>11} | {res_tag:>15} | {ne:>5} | {Path(fig).name}")
    print(f"\n  ALL counts + top-5 faithful: {'YES' if all_ok else 'NO'}")
    print("STOP — inspection only, no logic changes.")


if __name__ == "__main__":
    main()
