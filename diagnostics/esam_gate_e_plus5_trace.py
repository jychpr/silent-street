"""
Gate E +5 NOVEL DIAGNOSTIC — inspection only. Mechanism of the 30->35 novel gain.

For each novel GT instance newly covered after USR (Gate D conv: entity tight box,
box IoU>=0.5), classify the covering post-USR entity as:
  (a) USR-ADDED  — index >= len(EMR before)  (an appended M_A mask)
  (b) BOX-GROWTH — index <  len(EMR before)  (a pre-existing EMR entity whose box
                   grew via Eq.10 fusion `entity |= M_A`)
Index alignment is exact because esam_gate_e builds after = [copy(before)...] then
appends; fusion edits in place. We CONFIRM (b) via mask-IoU(after[j], before[j]).

Also checks the reverse (covered by EMR, LOST after USR) so gross gain - gross loss
reconciles to +5. No tuning, no Gate D re-eval beyond this trace.
"""
import json
import os
import sys

import numpy as np
from pycocotools import mask as maskUtils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_d_eval import NOVEL, DIAG, ROOT, iou_matrix, mask_to_box  # noqa: E402

OUT = f"{ROOT}/diagnostics/output/esam_gateE"
IOU = 0.5


def box_iou(a, b):
    return float(iou_matrix([a], [b])[0, 0])


def ann_to_mask(seg, h, w):
    if isinstance(seg, list):
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    elif isinstance(seg["counts"], list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(bool)


def main():
    full = json.load(open(f"{ROOT}/data/Annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in full["categories"]}
    meta = {im["id"]: im for im in full["images"] if im["id"] in set(DIAG)}
    diag = set(DIAG)
    novel_gt = {iid: [] for iid in DIAG}
    for a in full["annotations"]:
        if a["image_id"] in diag and cat[a["category_id"]] in NOVEL:
            x, y, w, h = a["bbox"]
            novel_gt[a["image_id"]].append(
                {"box": [x, y, x + w, y + h], "name": cat[a["category_id"]],
                 "seg": a["segmentation"], "iscrowd": a["iscrowd"]})
    del full

    gained, lost = [], []
    for iid in DIAG:
        gts = novel_gt[iid]
        if not gts:
            continue
        H, W = meta[iid]["height"], meta[iid]["width"]
        npz = np.load(f"{OUT}/{iid}_entities.npz")
        before, after = npz["before"], npz["after"]   # (Nb,H,W),(Na,H,W) bool
        Nb = before.shape[0]
        bbox = [mask_to_box(before[j]) for j in range(before.shape[0])]
        abox = [mask_to_box(after[j]) for j in range(after.shape[0])]

        for gi, g in enumerate(gts):
            gb = g["box"]
            # best before / after entity by box IoU
            bj, bi = -1, 0.0
            for j, bx in enumerate(bbox):
                if bx is None:
                    continue
                io = box_iou(gb, bx)
                if io > bi:
                    bi, bj = io, j
            aj, ai = -1, 0.0
            for j, bx in enumerate(abox):
                if bx is None:
                    continue
                io = box_iou(gb, bx)
                if io > ai:
                    ai, aj = io, j
            cov_b, cov_a = bi >= IOU, ai >= IOU
            rec = dict(iid=iid, name=g["name"], gi=gi, gbox=gb, H=H, W=W,
                       before_bestj=bj, before_iou=bi, after_bestj=aj, after_iou=ai)
            if cov_a and not cov_b:
                gained.append(rec)
            elif cov_b and not cov_a:
                lost.append(rec)

    print("=" * 88)
    print(f"GAINED novel instances (covered AFTER USR, NOT before): {len(gained)}")
    print(f"LOST   novel instances (covered before, NOT after):     {len(lost)}")
    print(f"NET reconciliation: +{len(gained)} - {len(lost)} = {len(gained)-len(lost)}  (expect +5)")
    print("=" * 88)

    n_a = n_b = 0
    for r in gained:
        iid, aj, Nb_ = r["iid"], r["after_bestj"], None
        npz = np.load(f"{OUT}/{iid}_entities.npz")
        before, after = npz["before"], npz["after"]
        Nb_ = before.shape[0]
        is_added = aj >= Nb_
        print(f"\n--- {iid}  {r['name']}  (GT#{r['gi']})  GTbox={[round(v,1) for v in r['gbox']]}")
        print(f"    EMR best entity IoU={r['before_iou']:.3f} (<0.5, uncovered before)"
              f"   USR cover entity idx={aj}  IoU={r['after_iou']:.3f}")
        if is_added:
            n_a += 1
            # mask IoU of the added entity vs GT mask
            g = novel_gt[iid][r["gi"]]
            ma = after[aj]
            try:
                gm = ann_to_mask(g["seg"], r["H"], r["W"])
                inter = np.logical_and(ma, gm).sum()
                union = ma.sum() + gm.sum() - inter
                miou = inter / union if union else 0.0
            except Exception as e:
                miou = float("nan")
            print(f"    => (a) USR-ADDED entity (idx {aj} >= EMR N={Nb_}).  "
                  f"added-mask IoU with GT:  box={r['after_iou']:.3f}  mask={miou:.3f}")
            print(f"       {'GENUINELY SEGMENTED' if miou>=0.5 else 'box hits but mask weak'}")
        else:
            n_b += 1
            # box-growth: same index in before vs after
            bb = mask_to_box(before[aj])
            ab = mask_to_box(after[aj])
            iou_emr = box_iou(r["gbox"], bb) if bb else 0.0
            iou_usr = box_iou(r["gbox"], ab) if ab else 0.0
            # confirm it's the same EMR entity grown: before subset of after
            inter = np.logical_and(before[aj], after[aj]).sum()
            mi_self = inter / max(after[aj].sum(), 1)
            ba = (bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0
            aa = (ab[2]-ab[0])*(ab[3]-ab[1]) if ab else 0
            print(f"    => (b) BOX-GROWTH of pre-existing EMR entity idx {aj} (< N={Nb_}).")
            print(f"       EMR box={[round(v,1) for v in bb] if bb else None}  area={ba:.0f}")
            print(f"       USR box={[round(v,1) for v in ab] if ab else None}  area={aa:.0f}  "
                  f"(grew x{aa/max(ba,1):.2f})")
            print(f"       IoU-with-GT:  EMR={iou_emr:.3f}  ->  USR={iou_usr:.3f}  "
                  f"(crossed 0.5 by enlargement; before⊆after frac={mi_self:.2f})")

    print("\n" + "=" * 88)
    print(f"SUMMARY of the {len(gained)} gained:  (a) genuinely-newly-segmented = {n_a}   "
          f"(b) box-growth artifact = {n_b}")
    if lost:
        print(f"LOST (other direction): {len(lost)}")
        for r in lost:
            print(f"   {r['iid']} {r['name']} GT#{r['gi']}: EMR IoU={r['before_iou']:.3f}(>=0.5) "
                  f"-> USR IoU={r['after_iou']:.3f}(<0.5)")
    else:
        print("LOST (other direction): 0  — no novel instance was dropped by USR fusion.")
    print(f"NET = +{len(gained)} - {len(lost)} = {len(gained)-len(lost)}")
    print("STOP — trace only.")


if __name__ == "__main__":
    main()
