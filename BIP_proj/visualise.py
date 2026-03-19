"""
visualise.py — Show 5 example crops with annotated bounding boxes.
Run from BIP_proj/: python visualise.py
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from dataset import NeonTreeTiledDataset, collate_fn

PROJECT_ROOT = Path(__file__).parent

def denorm_rgb(tensor_chw: torch.Tensor) -> np.ndarray:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = tensor_chw[:3].permute(1, 2, 0).numpy()
    return np.clip(rgb * STD + MEAN, 0, 1)

dataset = NeonTreeTiledDataset(
    rgb_dir=PROJECT_ROOT / "training" / "RGB",
    chm_dir=PROJECT_ROOT / "training" / "CHM",
    ann_dir=PROJECT_ROOT / "annotations" / "annotations",
    crop_size=512,
    stride=256,
    skip_empty=True,
)

# Pick 5 random crops
indices = random.sample(range(len(dataset)), 5)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("5 random crops with annotated tree boxes", fontsize=13)

for ax, idx in zip(axes, indices):
    sample = dataset[idx]
    rgb    = denorm_rgb(sample['image'])   # (H, W, 3)
    boxes  = sample['boxes']              # (N, 4)
    n      = len(boxes)

    ax.imshow(rgb)

    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1, edgecolor='lime', facecolor='none'
        ))

    x0, y0, x1, y1 = sample['window']
    ax.set_title(f"crop [{x0},{y0}]\n{n} trees", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig("crops_preview.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → crops_preview.png")