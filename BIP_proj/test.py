"""
test.py — Evaluate a saved Faster R-CNN checkpoint on the held-out test set.

Metrics reported:
    mAP@0.50        (PASCAL VOC style)
    mAP@0.50:0.95   (COCO style)
    Precision / Recall / F1  at IoU=0.50, confidence=0.50

Usage:
    python test.py                          # uses checkpoints/best_model.pth
    python test.py --checkpoint my.pth      # custom checkpoint
    python test.py --conf 0.3 --iou 0.4     # tune thresholds

Requires:
    pip install torchmetrics
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from load_display_data import load_files
from dataset import NeonTreeTiledDataset, collate_fn


# ── Shared config (must match train.py) ───────────────────────────────────────

CFG = dict(
    rgb_dir        = "training/RGB",
    chm_dir        = "training/CHM",
    ann_dir        = "annotations/annotations",
    crop_size      = 512,
    stride         = 256,
    skip_empty     = True,
    num_classes    = 2,
    num_workers    = 4,
    batch_size     = 4,
    checkpoint_dir = "checkpoints",
    seed           = 42,
    val_fraction   = 0.10,
    test_fraction  = 0.10,
)


# ── Model builder (identical to train.py) ─────────────────────────────────────

def patch_first_conv(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d) and child.in_channels == 3:
            new_conv = torch.nn.Conv2d(
                in_channels  = 4,
                out_channels = child.out_channels,
                kernel_size  = child.kernel_size,
                stride       = child.stride,
                padding      = child.padding,
                bias         = child.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = child.weight
                new_conv.weight[:, 3:] = child.weight.mean(dim=1, keepdim=True)
            setattr(module, name, new_conv)
            return True
        if patch_first_conv(child):
            return True
    return False


def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    )
    patch_first_conv(model.backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(
        min_size   = 512,
        max_size   = 512,
        image_mean = [0.0, 0.0, 0.0, 0.0],
        image_std  = [1.0, 1.0, 1.0, 1.0],
    )
    return model


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(model, loader, device, conf_threshold: float, iou_threshold: float):
    """
    Run inference over the loader and compute:
        - mAP@0.50 and mAP@0.50:0.95  via torchmetrics
        - Precision, Recall, F1 at the given conf/iou thresholds
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=None)  # uses COCO default

    tp = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            images  = [img.to(device) for img in batch['image']]
            outputs = model(images)

            preds   = []
            targets = []

            for out, boxes_gt, labels_gt in zip(
                outputs, batch['boxes'], batch['labels']
            ):
                # Filter predictions by confidence
                keep  = out['scores'] >= conf_threshold
                p_boxes  = out['boxes'][keep].cpu()
                p_scores = out['scores'][keep].cpu()
                p_labels = out['labels'][keep].cpu()

                preds.append({
                    'boxes':  p_boxes,
                    'scores': p_scores,
                    'labels': p_labels,
                })
                targets.append({
                    'boxes':  boxes_gt.cpu(),
                    'labels': (labels_gt + 1).cpu(),   # 0=bg convention
                })

                # ── Simple TP/FP/FN at iou_threshold ──────────────────────────
                gt_boxes = boxes_gt.cpu()
                matched  = torch.zeros(len(gt_boxes), dtype=torch.bool)

                for pb in p_boxes:
                    if len(gt_boxes) == 0:
                        fp += 1
                        continue
                    ious = box_iou_single(pb, gt_boxes)
                    best_idx = ious.argmax().item()
                    if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                        tp += 1
                        matched[best_idx] = True
                    else:
                        fp += 1

                fn += (~matched).sum().item()

            metric.update(preds, targets)

    map_result = metric.compute()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        'mAP@0.50'      : map_result['map_50'].item(),
        'mAP@0.50:0.95' : map_result['map'].item(),
        'precision'     : precision,
        'recall'        : recall,
        'f1'            : f1,
        'tp': tp, 'fp': fp, 'fn': fn,
    }


def box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """IoU between one box (4,) and many boxes (N, 4)."""
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Faster R-CNN tree detector")
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--conf', type=float, default=0.50,
                        help='Confidence threshold for predictions (default: 0.50)')
    parser.add_argument('--iou',  type=float, default=0.50,
                        help='IoU threshold for TP/FP matching (default: 0.50)')
    args = parser.parse_args()

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_path = Path(args.checkpoint) if args.checkpoint \
                else ckpt_dir / 'best_model.pth'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Conf ≥ {args.conf}   IoU ≥ {args.iou}\n")

    # ── Reconstruct the exact same test split as training ─────────────────────
    load_files()

    dataset = NeonTreeTiledDataset(
        rgb_dir   = CFG['rgb_dir'],
        chm_dir   = CFG['chm_dir'],
        ann_dir   = CFG['ann_dir'],
        crop_size = CFG['crop_size'],
        stride    = CFG['stride'],
        skip_empty= CFG['skip_empty'],
    )

    test_indices_path = ckpt_dir / 'test_indices.pt'
    if test_indices_path.exists():
        # Use the saved indices so the split is byte-for-byte identical
        test_indices = torch.load(test_indices_path)
        test_ds      = Subset(dataset, test_indices)
        print(f"Loaded saved test split: {len(test_ds)} crops")
    else:
        # Fall back: re-derive with the same seed (reproducible)
        from torch.utils.data import random_split
        n_total = len(dataset)
        n_val   = int(n_total * CFG['val_fraction'])
        n_test  = int(n_total * CFG['test_fraction'])
        n_train = n_total - n_val - n_test
        generator = torch.Generator().manual_seed(CFG['seed'])
        _, _, test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )
        print(f"Re-derived test split (no saved indices found): {len(test_ds)} crops")

    test_loader = DataLoader(
        test_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = CFG['num_workers'],
        pin_memory  = torch.cuda.is_available(),
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(CFG['num_classes']).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    trained_epoch = ckpt.get('epoch', '?')
    best_val_loss = ckpt.get('val_loss', '?')
    print(f"Loaded weights from epoch {trained_epoch}  "
          f"(val loss at save: {best_val_loss:.4f})\n")

    # ── Run evaluation ────────────────────────────────────────────────────────
    print("Running evaluation …")
    results = evaluate(model, test_loader, device, args.conf, args.iou)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  mAP @ 0.50        : {results['mAP@0.50']:.4f}")
    print(f"  mAP @ 0.50:0.95   : {results['mAP@0.50:0.95']:.4f}")
    print(f"{'─'*40}")
    print(f"  Precision         : {results['precision']:.4f}")
    print(f"  Recall            : {results['recall']:.4f}")
    print(f"  F1                : {results['f1']:.4f}")
    print(f"{'─'*40}")
    print(f"  TP: {results['tp']}   FP: {results['fp']}   FN: {results['fn']}")
    print(f"{'─'*40}\n")


if __name__ == '__main__':
    main()