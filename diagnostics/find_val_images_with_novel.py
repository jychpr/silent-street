"""
find_val_images_with_novel.py
==============================
Reads a COCO-format annotation JSON and prints the top N images by
novel-class GT count. Useful for picking stratified val images for
the side-by-side comparison.

Usage:
    python diagnostics/find_val_images_with_novel.py \
        --coco_ann data/Annotations/instances_val2017_basetarget.json \
        --top 20
"""

import argparse
import json
from collections import Counter


# Default: the 17 OV-COCO novel category IDs
_DEFAULT_NOVEL_IDS = '32,36,5,6,41,76,47,17,18,49,81,21,22,87,28,61,63'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--coco_ann', required=True,
                   help='Path to COCO-format annotation JSON')
    p.add_argument('--top', type=int, default=20,
                   help='Number of images to list')
    p.add_argument('--novel_ids', default=_DEFAULT_NOVEL_IDS,
                   help='Comma-separated category IDs to count as novel')
    return p.parse_args()


def main():
    args = parse_args()
    novel_ids = {int(x.strip()) for x in args.novel_ids.split(',')}

    with open(args.coco_ann) as f:
        data = json.load(f)

    img_info = {img['id']: img['file_name'] for img in data.get('images', [])}

    counts = Counter()
    for ann in data.get('annotations', []):
        if ann.get('category_id') in novel_ids:
            counts[ann['image_id']] += 1

    top = counts.most_common(args.top)
    for image_id, count in top:
        fname = img_info.get(image_id, '?')
        print(f"{image_id}: {count} novel objects (file: {fname})")


if __name__ == '__main__':
    main()
