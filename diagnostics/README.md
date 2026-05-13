# diagnostics/

This directory contains **pre-integration measurement scripts** that do not depend on the OV-DQUO model or training code.

## Purpose

Before wiring a proposal generator into the detection pipeline, these scripts characterise its raw output on COCO so we know what signal we are working with:

- How many proposals does it produce per image?
- What is the score distribution?
- What is the mask/box area distribution?

This lets us set sensible thresholds and evaluate whether a generator is worth integrating — without touching any model weights or training configuration.

## Scripts

| Script | Description |
|---|---|
| `fastsam_proposals.py` | Runs FastSAM in *everything* mode over a COCO split and saves per-image boxes, scores, and mask areas to JSON. |
| `fastsam_recall_precision.py` | Reads the proposals JSON and a COCO annotation file; computes proposal recall and precision broken down by object scale (small/medium/large) and OV-COCO base/novel split; writes a Markdown report. |

## Non-goals

- These scripts are **not** part of the model.
- They do **not** import or modify anything under `src/`, `models/`, `engine.py`, or `config/`.
- Output JSON files are gitignored; only the scripts and this README are tracked.

## Usage

```bash
# Step 1 — generate proposals
python diagnostics/fastsam_proposals.py \
    --coco-root data/ \
    --split val2017 \
    --num-images 5000 \
    --output diagnostics/output/val2017_proposals.json \
    --fastsam-weights FastSAM-x.pt

# Step 2 — measure recall & precision (use basetarget ann for novel recall)
python diagnostics/fastsam_recall_precision.py \
    --proposals diagnostics/output/val2017_proposals.json \
    --coco-ann data/Annotations/instances_val2017_basetarget.json \
    --split-name val5000 \
    --output diagnostics/output/val5000_recall_report.md
```

See `--help` on each script for all options.
