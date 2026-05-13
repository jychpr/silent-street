# Add to tools/analyze_pseudo_labels.py or run inline
import json
import numpy as np
from PIL import Image
from pathlib import Path

# For one of the FastSAM JSONs
# ann = json.load(open("ow_labels/OW_COCO_FASTSAM_K5_conf010.json"))
ann = json.load(open("ow_labels/OW_COCO_FASTSAM_K5_conf010_filtered.json"))

# Build image_id -> (width, height) map from COCO base annotations
coco = json.load(open("data/Annotations/instances_train2017_base.json"))
img_dims = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

# Compute box-area / image-area ratio for every annotation
ratios = []
aspects = []
for a in ann:
    iw, ih = img_dims[a["image_id"]]
    bw, bh = a["bbox"][2], a["bbox"][3]
    ratios.append((bw * bh) / (iw * ih))
    aspects.append(max(bw / bh, bh / bw))

ratios = np.array(ratios)
aspects = np.array(aspects)

print(f"Box/image area ratio:")
print(f"  > 0.4 (full-image-ish): {(ratios > 0.4).mean()*100:.2f}% of all boxes")
print(f"  > 0.6:                  {(ratios > 0.6).mean()*100:.2f}%")
print(f"  > 0.8:                  {(ratios > 0.8).mean()*100:.2f}%")
print(f"Aspect ratio (max(w/h, h/w)):")
print(f"  > 5 (long thin):  {(aspects > 5).mean()*100:.2f}% of all boxes")
print(f"  > 8:              {(aspects > 8).mean()*100:.2f}%")