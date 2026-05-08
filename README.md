# silent-street

# Initial Setup
## Environment and Dependencies Setup
1. Clone the repository and go to directory
```bash
git clone git@github.com:jychpr/silent-street.git
cd silent-street/
```
2. Create environment
```bash
conda create -n SS python=3.10 -y
conda activate SS
```
3. Run this command step by step for smooth installation of dependencies
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
```bash
# Meta foundational libraries
pip install fvcore iopath omegaconf cloudpickle black hydra-core tensorboard
# Standard vision/evaluation stack
pip install lvis pycocotools scipy shapely pandas opencv-python tqdm timm submitit einops transformers open_clip_torch torchmetrics mmcv==1.7.1 termcolor yapf==0.32.0
```
The original OV-DQUO authors used a deprecated PyTorch C++ API (value.type().is_cuda()) that has been entirely removed from modern PyTorch. Patch their source code before compiling.
```bash
# 1. Fix CUDA assertions
find models/ops/src -type f -exec sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' {} +

# 2. Fix legacy type dispatch macros
find models/ops/src -type f -exec sed -i 's/value\.type()/value.scalar_type()/g' {} +
```
```bash
# Target RTX 5090 natively
export TORCH_CUDA_ARCH_LIST="12.0"

# Build Custom Deformable Attention
cd models/ops
rm -rf build/ dist/
sh make.sh
cd ../../

# Build Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation --force-reinstall --no-deps
```
4. Proposed Method Dependencies

Additional dependencies
```bash
pip install openpyxl ultralytics
```

## Data Preparation & Checkpoints
Ensure your directory structure aligns with standard COCO formats. Pre-trained base weights (OVDQUO_RN50_COCO.pth) must be placed in the /ckpt/ directory prior to evaluation.

And please make sure that you have these in your folders, if not, refer to the original OV-DQUO repo to download the weights:
- ckpt/OVDQUO_RN50_COCO.pth
- ckpt/OVDQUO_RN50x4_COCO.pth
- pretrained/region_prompt_R50.pth
- pretrained/region_prompt_R50x4.pth
- ow_labels/OW_COCO_R1.json
- ow_labels/OW_COCO_R2.json
- ow_labels/OW_COCO_R3.json
- ow_labels/OW_LVIS_R3.json

These for the dataset ov-coco, coco2017 images, and also the annotations of original coco and ov-coco annotations
- data/Anotations/captions_train2017.json
- data/Anotations/captions_val2017.json
- data/Anotations/image_info_test-dev2017.json
- data/Anotations/image_info_test2017.json
- data/Anotations/instances_train2017_base.json
- data/Anotations/instances_train2017.json
- data/Anotations/instances_val2017_basetarget.json
- data/Anotations/instances_val2017.json
- data/Anotations/person_keypoints_train2017.json
- data/Anotations/person_keypoints_val2017.json
- data/Images/train2017
- data/Images/val2017
- data/Images/test2017

If the above hasn't been done, please check the following section

## Downloading Required Files (CLI / SSH)

This section is for headless or SSH environments. Install gdown first:

```bash
pip install gdown
```

Then run the following commands from the repo root to download and place all required files.

**OV-DQUO Checkpoints**
```bash
mkdir -p ckpt
gdown "https://drive.google.com/uc?id=17Nlo0V4jrJz0bNvivfFXcOcaYZq-Up3x" -O ckpt/OVDQUO_RN50_COCO.pth
gdown "https://drive.google.com/uc?id=1bDxIj1spUmqrMRNHGzK5TZd9uhL9T1KG" -O ckpt/OVDQUO_RN50x4_COCO.pth
```

**Pretrained Region-Prompt Weights**
```bash
mkdir -p pretrained
gdown --folder "https://drive.google.com/drive/folders/17mi8O1YW6dl8TRkwectHRoC8xbK5sLMw" -O pretrained/
```

**OW Label JSONs**
```bash
mkdir -p ow_labels
gdown --folder "https://drive.google.com/drive/folders/1j-i6BkbsHvD_pNXVZRQ6fmAYOWnF4Ao4" -O ow_labels/
```

**OV-COCO Annotations**
```bash
mkdir -p data/Anotations
gdown --folder "https://drive.google.com/drive/folders/1Jgkpoz_ILJRI4xRJydi7dQfFjwtAFbef" -O data/Anotations/
```

> ⚠ **Disk space:** COCO 2017 requires approximately 26 GB of free disk space (19 GB train images, 1 GB val/test images, ~1 GB annotations). Verify available space with `df -h .` before starting.

**COCO 2017 Original Dataset (Images + Annotations)**
```bash
mkdir -p data/Images data/Anotations

# Images
wget -P data/Images/ http://images.cocodataset.org/zips/train2017.zip
wget -P data/Images/ http://images.cocodataset.org/zips/val2017.zip
wget -P data/Images/ http://images.cocodataset.org/zips/test2017.zip

# Annotations
wget -P data/Anotations/ http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -P data/Anotations/ http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -P data/Anotations/ http://images.cocodataset.org/annotations/image_info_test-dev2017.zip

# Unzip
unzip data/Images/train2017.zip  -d data/Images/
unzip data/Images/val2017.zip    -d data/Images/
unzip data/Images/test2017.zip   -d data/Images/
unzip data/Anotations/annotations_trainval2017.zip -d data/Anotations/
unzip data/Anotations/image_info_test2017.zip      -d data/Anotations/
unzip data/Anotations/image_info_test-dev2017.zip  -d data/Anotations/

# Optional: remove zips to save space
rm data/Images/*.zip data/Anotations/*.zip
```

> **Note on gdown folder downloads:** if you hit a Google Drive quota or rate-limit error, add `--remaining-ok` to skip already-downloaded files. For very large folders, consider using `rclone` with a Google service account for more reliable transfers.