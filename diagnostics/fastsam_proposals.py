"""
Measure FastSAM proposal quality on COCO.

Runs FastSAM in everything mode over a sampled subset of a COCO split and
writes per-image boxes, confidence scores, and mask areas to a JSON file.
This script is fully standalone — it does not import any OV-DQUO code.
"""

import argparse
import json
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FastSAM proposal quality measurement on COCO")
    p.add_argument("--coco-root", required=True, help="COCO root dir (contains Annotations/ and Images/)")
    p.add_argument("--split", default="val2017", choices=["train2017", "val2017"], help="COCO split")
    p.add_argument("--num-images", type=int, default=5000, help="Number of images to sample")
    p.add_argument("--output", required=True, help="Path to output .json file")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt", help="FastSAM weights file or path")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p.add_argument("--device", default="cuda:0", help="Device (cuda:0, cpu, …)")
    return p.parse_args()


def load_coco_image_index(coco_root: str, split: str) -> dict:
    """Return {image_id: image_info_dict} from the instances annotation file."""
    ann_path = os.path.join(coco_root, "Annotations", f"instances_{split}.json")
    log.info(f"Loading COCO annotations from {ann_path}")
    with open(ann_path) as f:
        data = json.load(f)
    return {img["id"]: img for img in data["images"]}


def main():
    args = parse_args()

    from ultralytics import FastSAM  # noqa: PLC0415 — deferred so --help works without GPU

    image_index = load_coco_image_index(args.coco_root, args.split)
    all_ids = sorted(image_index.keys())

    rng = random.Random(42)
    sampled_ids = rng.sample(all_ids, min(args.num_images, len(all_ids)))
    log.info(f"Sampled {len(sampled_ids)} images from {args.split}")

    model = FastSAM(args.fastsam_weights)
    log.info(f"FastSAM loaded from {args.fastsam_weights}")

    fastsam_config = {
        "weights": args.fastsam_weights,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
    }

    proposals = []
    images_dir = os.path.join(args.coco_root, "Images", args.split)

    for i, image_id in enumerate(sampled_ids):
        if i > 0 and i % 100 == 0:
            log.info(f"Progress: {i}/{len(sampled_ids)}")

        info = image_index[image_id]
        img_path = os.path.join(images_dir, info["file_name"])

        results = model(
            img_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )

        boxes_xyxy = []
        scores = []
        mask_areas = []

        if results and results[0].boxes is not None:
            res = results[0]
            # boxes are already in original image coordinates when retina_masks=True
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_xyxy.append([x1, y1, x2, y2])
                scores.append(float(box.conf[0]))

            if res.masks is not None:
                for mask in res.masks.data:
                    mask_areas.append(int(mask.sum().item()))
            else:
                mask_areas = [0] * len(boxes_xyxy)

        proposals.append({
            "image_id": image_id,
            "image_width": info["width"],
            "image_height": info["height"],
            "boxes": boxes_xyxy,
            "scores": scores,
            "mask_areas": mask_areas,
        })

    output = {
        "split": args.split,
        "num_images": len(proposals),
        "fastsam_config": fastsam_config,
        "proposals": proposals,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)
    log.info(f"Wrote {len(proposals)} proposal entries to {args.output}")


if __name__ == "__main__":
    main()
