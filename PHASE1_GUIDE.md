# Phase 1: Baseline Model - Quick Start Guide

## Overview
Train a model on full objects and evaluate on progressive occlusion (0%, 25%, 50%, 75%).

## Step-by-Step Workflow

### Step 1: Split Dataset
Split your dataset into train/val/test (70/15/15):

```bash
make split-dataset DATASET=datasets/credit-cards-coco
```

**Output:**
- `datasets/credit-cards-coco_split/train/` (210 images)
- `datasets/credit-cards-coco_split/val/` (45 images)
- `datasets/credit-cards-coco_split/test/` (45 images)

### Step 2: Generate Progressive Occlusion Test Sets
Create test sets with 0%, 25%, 50%, 75% occlusion:

```bash
make prepare-progressive-tests TEST_DATASET=datasets/credit-cards-coco_split/test
```

**Output:**
- `datasets/credit-cards-coco_split/test_occlusion_0/` (baseline, 0%)
- `datasets/credit-cards-coco_split/test_occlusion_25/` (25% occlusion)
- `datasets/credit-cards-coco_split/test_occlusion_50/` (50% occlusion)
- `datasets/credit-cards-coco_split/test_occlusion_75/` (75% occlusion)

### Step 3: Train Model
Train YOLOv8 model on full objects:

```bash
# Quick training (nano model, 50 epochs)
make train-model DATASET=datasets/credit-cards-coco_split/train MODEL_SIZE=n EPOCHS=50

# Full training (small model, 100 epochs)
make train-model DATASET=datasets/credit-cards-coco_split/train MODEL_SIZE=s EPOCHS=100
```

**Model Sizes:**
- `n` = nano (fastest, smallest)
- `s` = small (balanced)
- `m` = medium (better accuracy)
- `l` = large (best accuracy, slower)

**Output:**
- `models/credit_card_{size}/weights/best.pt` (best model)
- `models/credit_card_{size}/weights/last.pt` (last checkpoint)
- Training plots and metrics

### Step 4: Evaluate on Progressive Occlusion
Test model on all occlusion levels:

```bash
make evaluate-progressive \
    MODEL=models/credit_card_n/weights/best.pt \
    TEST_SETS=datasets/credit-cards-coco_split
```

**Output:**
- `outputs/progressive_evaluation/progressive_results.json` (metrics)
- `outputs/progressive_evaluation/progressive_results.csv` (table)
- `outputs/progressive_evaluation/progressive_occlusion_results.png` (plot)

## Quick Run (All Steps)

```bash
# 1. Split dataset
make split-dataset DATASET=datasets/credit-cards-coco

# 2. Prepare progressive tests
make prepare-progressive-tests TEST_DATASET=datasets/credit-cards-coco_split/test

# 3. Train model
make train-model DATASET=datasets/credit-cards-coco_split/train MODEL_SIZE=n EPOCHS=50

# 4. Evaluate
make evaluate-progressive \
    MODEL=models/credit_card_n/weights/best.pt \
    TEST_SETS=datasets/credit-cards-coco_split
```

## Expected Results

After evaluation, you'll get:
- Performance metrics (mAP, precision, recall, F1) for each occlusion level
- Visualization showing performance degradation
- CSV file with all metrics

## Troubleshooting

### Issue: "Annotation file not found"
- Make sure dataset path is correct
- Check that `_annotations.coco.json` exists in train folder

### Issue: "CUDA out of memory"
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `MODEL_SIZE=n`

### Issue: Training takes too long
- Use nano model: `MODEL_SIZE=n`
- Reduce epochs: `EPOCHS=50`
- Reduce image size: `--imgsz 416`

## Next Steps After Phase 1

1. **Analyze Results**: Review performance degradation curve
2. **Phase 2** (Optional): Train on full + occluded objects
3. **Compare**: See if training on occlusion improves robustness
4. **Presentation**: Use results for your Thursday presentation

## Time Estimates

- Dataset splitting: < 1 minute
- Generate progressive tests: ~5-10 minutes
- Training (nano, 50 epochs): ~30-60 minutes (CPU) / ~5-10 minutes (GPU)
- Evaluation: ~5-10 minutes

**Total: ~1-2 hours** (depending on hardware)

