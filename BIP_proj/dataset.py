"""
Each large image is divided into (CROP_SIZE × CROP_SIZE) crops with a
configurable overlap (stride). For every crop we:
  1. Cut the same window out of RGB and CHM (after upsampling CHM to RGB res)
  2. Keep only boxes whose centre falls inside the crop window
  3. Clip those boxes to the crop boundary
  4. Discard boxes that become too small after clipping (min_box_side pixels)
  5. Shift box coordinates so they are relative to the crop origin
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset



# Normalisation constants  (update with dataset-wide stats if desired)
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CHM_MEAN = 5.0
CHM_STD  = 5.0



# Low-level I/O
def load_rgb(path: str | Path) -> np.ndarray:
    """Load RGB .tif → uint8 (H, W, 3)."""
    return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)


def load_chm(path: str | Path) -> np.ndarray:
    """Load single-band CHM .tif → float32 (H, W), nodata → 0."""
    arr = np.array(Image.open(path), dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None)


def resize_numpy(arr: np.ndarray, h: int, w: int,
                 resample=Image.BILINEAR) -> np.ndarray:
    """Resize a numpy array (H,W) or (H,W,C) to (h,w[,C])."""
    dtype = arr.dtype
    if arr.ndim == 2:
        out = np.array(Image.fromarray(arr, mode='F').resize((w, h), resample),
                       dtype=dtype)
    else:
        out = np.array(Image.fromarray(arr).resize((w, h), resample),
                       dtype=dtype)
    return out



# XML parsing
def parse_annotation(xml_path: str | Path) -> dict:
    """
    Parse Pascal-VOC-style XML.
    Returns dict with keys: filename, orig_w, orig_h, boxes (N,4), labels.
    Handles the non-standard <n> tag used in this dataset.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    orig_w = int(size.findtext('width'))
    orig_h = int(size.findtext('height'))

    boxes, labels = [], []
    for obj in root.findall('object'):
        label = obj.findtext('name') or obj.findtext('n') or 'Tree'
        bb = obj.find('bndbox')
        boxes.append([
            float(bb.findtext('xmin')), float(bb.findtext('ymin')),
            float(bb.findtext('xmax')), float(bb.findtext('ymax')),
        ])
        labels.append(label)

    boxes_arr = np.array(boxes, dtype=np.float32) if boxes else \
                np.zeros((0, 4), dtype=np.float32)
    return {
        'filename': root.findtext('filename', ''),
        'orig_w': orig_w, 'orig_h': orig_h,
        'boxes': boxes_arr, 'labels': labels,
    }


# Box clipping helpers

def clip_boxes_to_crop(boxes: np.ndarray, labels: list,
                       x0: int, y0: int, x1: int, y1: int,
                       min_box_side: float = 4.0,
                       centre_must_be_inside: bool = True
                       ) -> tuple[np.ndarray, list]:
    """
    Given global-coordinate boxes, keep those relevant to crop [x0,y0,x1,y1],
    clip them to the crop boundary, and shift to crop-local coordinates.

    centre_must_be_inside=True  → only keep boxes whose centre is in the crop
                                   (avoids double-counting at tile borders)
    centre_must_be_inside=False → keep any box that overlaps the crop at all

    min_box_side: after clipping, discard boxes narrower/shorter than this (px).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), []

    if centre_must_be_inside:
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        mask = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    else:
        mask = (boxes[:, 2] > x0) & (boxes[:, 0] < x1) & \
               (boxes[:, 3] > y0) & (boxes[:, 1] < y1)

    boxes  = boxes[mask]
    labels = [l for l, m in zip(labels, mask) if m]

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), []

    # Clip to crop window then shift to local coordinates
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(boxes[:, 0], x0, x1) - x0  # xmin
    clipped[:, 1] = np.clip(boxes[:, 1], y0, y1) - y0  # ymin
    clipped[:, 2] = np.clip(boxes[:, 2], x0, x1) - x0  # xmax
    clipped[:, 3] = np.clip(boxes[:, 3], y0, y1) - y0  # ymax

    # Drop boxes that shrank too much after clipping
    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    valid  = (w >= min_box_side) & (h >= min_box_side)
    clipped = clipped[valid]
    labels  = [l for l, v in zip(labels, valid) if v]

    return clipped.astype(np.float32), labels


# ──────────────────────────────────────────────────────────────────────────────
# Tile index builder
# ──────────────────────────────────────────────────────────────────────────────

def build_tile_index(image_h: int, image_w: int,
                     crop_size: int, stride: int
                     ) -> list[tuple[int, int, int, int]]:
    """
    Return (x0, y0, x1, y1) crop windows covering the full image.
    The last column/row snaps to the image boundary so every pixel is included.
    """
    windows = []
    y = 0
    while True:
        y1 = min(y + crop_size, image_h)
        y0 = y1 - crop_size
        x = 0
        while True:
            x1 = min(x + crop_size, image_w)
            x0 = x1 - crop_size
            windows.append((x0, y0, x1, y1))
            if x1 == image_w:
                break
            x += stride
        if y1 == image_h:
            break
        y += stride
    return windows


# Normalisation

def normalise_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    return (rgb_uint8.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD


def normalise_chm(chm: np.ndarray) -> np.ndarray:
    return (chm - CHM_MEAN) / CHM_STD


# Main dataset
class NeonTreeTiledDataset(Dataset):
    """
    Sliding-window crop dataset for tree detection using RGB + CHM.

    Each item is a (crop_size × crop_size) patch from a large aerial tile,
    with bounding boxes clipped and shifted to patch-local coordinates.

    Parameters
    ──────────
    rgb_dir, chm_dir, ann_dir : paths to data folders
    crop_size : spatial size of each crop in pixels (default 512)
    stride    : step between crop origins; stride < crop_size means overlapping
                crops. Use stride = crop_size for non-overlapping tiles.
                Use stride = crop_size // 2 for 50% overlap (more training data,
                trees near borders appear in multiple crops).
    min_box_side : discard boxes smaller than this after clipping (px)
    skip_empty   : if True, crops with zero boxes are excluded from the dataset
                   (speeds up training; set False if you want hard negatives)
    centre_must_be_inside : see clip_boxes_to_crop docstring
    transforms   : optional callable applied to the output dict
    """

    CLASS_NAMES = ['Tree']

    def __init__(
        self,
        rgb_dir:  str | Path,
        chm_dir:  str | Path,
        ann_dir:  str | Path,
        crop_size: int   = 512,
        stride:    int   = 256,
        min_box_side: float = 8.0,
        skip_empty:   bool  = True,
        centre_must_be_inside: bool = True,
        transforms = None,
    ):
        self.rgb_dir   = Path(rgb_dir)
        self.chm_dir   = Path(chm_dir)
        self.ann_dir   = Path(ann_dir)
        self.crop_size = crop_size
        self.stride    = stride
        self.min_box_side = min_box_side
        self.skip_empty   = skip_empty
        self.centre_must_be_inside = centre_must_be_inside
        self.transforms = transforms

        self._index = self._build_index()
        n_tiles = len(set(e['stem'] for e in self._index))
        print(f'  → {len(self._index)} crops from {n_tiles} tiles '
              f'(crop={crop_size}px, stride={stride}px, '
              f'skip_empty={skip_empty})')

    def _build_index(self) -> list[dict]:
        index = []
        for xml_path in sorted(self.ann_dir.glob('*.xml')):
            stem = xml_path.stem
            rgb_path = self.rgb_dir / f'{stem}.tif'
            if not rgb_path.exists():
                continue

            chm_path = self.chm_dir / f'{stem}_CHM.tif'
            if not chm_path.exists():
                chm_path = self.chm_dir / f'{stem}.tif'
                if not chm_path.exists():
                    chm_path = None

            ann = parse_annotation(xml_path)

            # Get real image dimensions cheaply without decoding pixels
            with Image.open(rgb_path) as im:
                real_w, real_h = im.size   # PIL returns (width, height)

            # Scale annotation boxes if XML size doesn't match the file
            boxes  = ann['boxes'].copy()
            labels = list(ann['labels'])
            if boxes.shape[0] > 0 and \
               (ann['orig_w'] != real_w or ann['orig_h'] != real_h):
                sx = real_w / ann['orig_w']
                sy = real_h / ann['orig_h']
                boxes[:, 0::2] *= sx
                boxes[:, 1::2] *= sy

            windows = build_tile_index(real_h, real_w,
                                       self.crop_size, self.stride)

            for win in windows:
                x0, y0, x1, y1 = win
                crop_boxes, crop_labels = clip_boxes_to_crop(
                    boxes, labels, x0, y0, x1, y1,
                    self.min_box_side, self.centre_must_be_inside
                )
                if self.skip_empty and len(crop_boxes) == 0:
                    continue
                index.append({
                    'rgb_path':  rgb_path,
                    'chm_path':  chm_path,
                    'image_h':   real_h,
                    'image_w':   real_w,
                    'window':    win,
                    'boxes':     crop_boxes,   # already crop-local coords
                    'labels':    crop_labels,
                    'stem':      stem,
                })
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        entry = self._index[idx]
        x0, y0, x1, y1 = entry['window']
        cs = self.crop_size

        #Load RGB crop
        with Image.open(entry['rgb_path']) as im:
            rgb_crop = np.array(im.convert('RGB'), dtype=np.uint8)[y0:y1, x0:x1]

        # ── 2. Load CHM crop (extract at CHM resolution, upsample to crop_size)
        if entry['chm_path'] is not None:
            with Image.open(entry['chm_path']) as im:
                chm_full = np.array(im, dtype=np.float32)
            chm_full = np.nan_to_num(chm_full, nan=0.0, posinf=0.0, neginf=0.0)
            chm_full = np.clip(chm_full, 0.0, None)

            chm_h, chm_w = chm_full.shape
            img_h = entry['image_h']
            img_w = entry['image_w']

            if chm_h != img_h or chm_w != img_w:
                # Map the crop window into CHM pixel space
                scale_x = chm_w / img_w
                scale_y = chm_h / img_h
                cx0 = int(x0 * scale_x)
                cy0 = int(y0 * scale_y)
                cx1 = max(int(x1 * scale_x), cx0 + 1)
                cy1 = max(int(y1 * scale_y), cy0 + 1)
                chm_patch = chm_full[cy0:cy1, cx0:cx1]
                chm_crop  = resize_numpy(chm_patch, cs, cs, Image.BILINEAR)
            else:
                chm_crop = chm_full[y0:y1, x0:x1]
        else:
            chm_crop = np.zeros((cs, cs), dtype=np.float32)

        # 3. Normalise
        rgb_norm = normalise_rgb(rgb_crop)              # (H, W, 3) float32
        chm_norm = normalise_chm(chm_crop)[..., None]  # (H, W, 1) float32

        #4. Fuse → (4, H, W) tensor 
        fused = np.concatenate([rgb_norm, chm_norm], axis=-1)
        image_tensor = torch.from_numpy(fused.transpose(2, 0, 1))

        # 5. Boxes & labels to tensors
        boxes_tensor  = torch.from_numpy(entry['boxes'])
        label_idx     = [self.CLASS_NAMES.index(l)
                         if l in self.CLASS_NAMES else 0
                         for l in entry['labels']]
        labels_tensor = torch.tensor(label_idx, dtype=torch.int64)

        sample = {
            'image':   image_tensor,    # float32 (4, crop_size, crop_size)
            'boxes':   boxes_tensor,    # float32 (N, 4) — crop-local coords
            'labels':  labels_tensor,   # int64   (N,)
            'stem':    entry['stem'],
            'window':  entry['window'],
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


# DataLoader collate
def collate_fn(batch: list[dict]) -> dict:
    """Stack images; keep boxes/labels as lists (variable N per crop)."""
    return {
        'image':  torch.stack([s['image']  for s in batch]),
        'boxes':  [s['boxes']  for s in batch],
        'labels': [s['labels'] for s in batch],
        'stem':   [s['stem']   for s in batch],
        'window': [s['window'] for s in batch],
    }

