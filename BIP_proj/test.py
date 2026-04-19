
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from load_display_data import load_files
from dataset import NeonTreeTiledDataset, collate_fn
from train import patch_first_conv


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

CONF_THRESHOLD = 0.50
IOU_THRESHOLD  = 0.50



def build_model(num_classes: int) -> torch.nn.Module:
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
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


def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.50])

    tp = fp = fn = 0
    tile_results = []
    stem_tp      = {}

    with torch.no_grad():
        for batch in loader:
            images  = [img.to(device) for img in batch['image']]
            outputs = model(images)

            preds   = []
            targets = []

            for out, boxes_gt, labels_gt, stem in zip(
                outputs, batch['boxes'], batch['labels'], batch['stem']
            ):
                keep     = out['scores'] >= CONF_THRESHOLD
                p_boxes  = out['boxes'][keep].cpu()
                p_scores = out['scores'][keep].cpu()
                p_labels = out['labels'][keep].cpu()

                preds.append({'boxes': p_boxes, 'scores': p_scores, 'labels': p_labels})
                targets.append({'boxes': boxes_gt.cpu(), 'labels': (labels_gt + 1).cpu()})

                gt_boxes = boxes_gt.cpu()
                matched  = torch.zeros(len(gt_boxes), dtype=torch.bool)

                crop_has_tp = False
                for pb in p_boxes:
                    if len(gt_boxes) == 0:
                        fp += 1
                        continue
                    ious     = box_iou_single(pb, gt_boxes)
                    best_idx = ious.argmax().item()
                    if ious[best_idx] >= IOU_THRESHOLD and not matched[best_idx]:
                        tp += 1
                        matched[best_idx] = True
                        crop_has_tp = True
                    else:
                        fp += 1

                fn += (~matched).sum().item()

                tile_results.append({'stem': stem, 'had_tp': crop_has_tp, 'n_gt': len(gt_boxes)})

                if stem not in stem_tp:
                    stem_tp[stem] = False
                if crop_has_tp:
                    stem_tp[stem] = True

            metric.update(preds, targets)

    map_result = metric.compute()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    n_crops             = len(tile_results)
    crops_with_tp       = sum(1 for t in tile_results if t['had_tp'])
    crops_with_gt       = [t for t in tile_results if t['n_gt'] > 0]
    n_crops_with_gt     = len(crops_with_gt)
    crops_with_gt_tp    = sum(1 for t in crops_with_gt if t['had_tp'])
    crop_detect_rate    = crops_with_tp / n_crops if n_crops > 0 else 0.0
    crop_detect_rate_gt = crops_with_gt_tp / n_crops_with_gt if n_crops_with_gt > 0 else 0.0

    n_stems          = len(stem_tp)
    stems_with_tp    = sum(1 for v in stem_tp.values() if v)
    stem_detect_rate = stems_with_tp / n_stems if n_stems > 0 else 0.0

    return {
        'mAP@0.50'           : map_result['map_50'].item(),
        'precision'          : precision,
        'recall'             : recall,
        'f1'                 : f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_crops'            : n_crops,
        'crops_with_tp'      : crops_with_tp,
        'crop_detect_rate'   : crop_detect_rate,
        'crops_with_gt_tp'   : crops_with_gt_tp,
        'crop_detect_rate_gt': crop_detect_rate_gt,
        'n_crops_with_gt'    : n_crops_with_gt,
        'n_stems'            : n_stems,
        'stems_with_tp'      : stems_with_tp,
        'stem_detect_rate'   : stem_detect_rate,
    }


def box_iou_single(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-6)

def main():
    ckpt_path = Path(CFG['checkpoint_dir']) / 'best_model.pth'

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}\n")

    load_files()

    dataset = NeonTreeTiledDataset(
        rgb_dir   = CFG['rgb_dir'],
        chm_dir   = CFG['chm_dir'],
        ann_dir   = CFG['ann_dir'],
        crop_size = CFG['crop_size'],
        stride    = CFG['stride'],
        skip_empty= CFG['skip_empty'],
    )

    test_indices_path = Path(CFG['checkpoint_dir']) / 'test_indices.pt'
    if test_indices_path.exists():
        test_indices = torch.load(test_indices_path)
        test_ds      = Subset(dataset, test_indices)
        print(f"Loaded saved test split: {len(test_ds)} crops")
    else:
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

    model = build_model(CFG['num_classes']).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    trained_epoch = ckpt.get('epoch', '?')
    best_val_loss = ckpt.get('val_loss', '?')
    print(f"Loaded weights from epoch {trained_epoch}  "
          f"(val loss at save: {best_val_loss:.4f})\n")

    results = evaluate(model, test_loader, device)

    print(f"  mAP @ 0.50        : {results['mAP@0.50']:.4f}")
    print(f"  Precision         : {results['precision']:.4f}")
    print(f"  Recall            : {results['recall']:.4f}")
    print(f"  F1                : {results['f1']:.4f}")
    print(f"  TP: {results['tp']}   FP: {results['fp']}   FN: {results['fn']}")
    print(f"\n  Per-crop detection rate")
    print(f"  Crops with trees  : {results['crops_with_gt_tp']:>4} / {results['n_crops_with_gt']:<4}"
          f"  ({results['crop_detect_rate_gt']*100:.1f}%)")
if __name__ == '__main__':
    main()