"""
train.py — Fine-tune Faster R-CNN for tree detection on 4-channel (RGB+CHM) input.

Usage:
    python train.py

Outputs (written to ./checkpoints/):
    best_model.pth      — weights with lowest validation loss
    last_model.pth      — weights at the final epoch
    training_log.csv    — per-epoch loss / LR history
"""

import copy
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from load_display_data import load_files
from dataset import NeonTreeTiledDataset, collate_fn




CFG = dict(
    # Data split Split 80 / 10 / 10  = train/ test / val
    val_fraction   = 0.10,
    test_fraction  = 0.10,
    seed           = 42,
    num_classes    = 2, 

    # Training
    num_epochs     = 30,
    batch_size     = 6,
    num_workers    = 6,           

    # Optimiser 
    lr             = 0.005,
    momentum       = 0.9,
    weight_decay   = 1e-4,

    # LR schedule
    lr_patience    = 4,           # epochs without improvement before LR drop
    lr_factor      = 0.5,         # multiply LR by this on plateau
    min_lr         = 1e-6,

    # Early stopping
    es_patience    = 5,          

    # Gradient clipping 
    grad_clip      = 5.0,
    
    # Misc
    checkpoint_dir = "checkpoints",
    amp            = True,        # mixed-precision (set False if no CUDA)
)


# ── 4-channel backbone ─────────────────────────────────────────────────────────

def patch_first_conv(module):
    """Recursively find and patch the first Conv2d(3→N) to accept 4 channels."""
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


def build_4ch_faster_rcnn(num_classes: int) -> torch.nn.Module:
    """
    Faster R-CNN with MobileNetV3 backbone patched for
    4-channel (RGB + CHM) input.
    """
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )

    patch_first_conv(model.backbone)

    #Replace classification head 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #4-channel identity transform, must be redefined for 4dimensions 
    model.transform = GeneralizedRCNNTransform(
        min_size   = 320,
        max_size   = 320,
        image_mean = [0.0, 0.0, 0.0, 0.0],
        image_std  = [1.0, 1.0, 1.0, 1.0],
    )

    return model


#Target formatting 

def batch_to_targets(boxes_list, labels_list, device):
    """
    Convert collated batch lists into the list-of-dicts format Faster R-CNN
    expects.  Labels are shifted +1 because 0 is reserved for background.
    """
    targets = []
    for boxes, labels in zip(boxes_list, labels_list):
        targets.append({
            'boxes':  boxes.to(device),
            'labels': (labels + 1).to(device),   # 0 = background
        })
    return targets


# ── One training epoch ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, scaler, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        images  = batch['image'].to(device)
        targets = batch_to_targets(batch['boxes'], batch['labels'], device)

        optimiser.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


#Validation loss

@torch.no_grad()
def evaluate_loss(model, loader, device):
    """
    Faster R-CNN only returns losses when in train() mode and given targets.
    We switch to train mode temporarily but disable dropout/BN updates via
    no_grad, which is fine for loss estimation.
    """
    model.train()          # needed to get loss dict
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        images  = batch['image'].to(device)
        targets = batch_to_targets(batch['boxes'], batch['labels'], device)

        loss_dict = model(images, targets)
        loss      = sum(loss_dict.values())

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


#Early stopping helper

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience   = patience
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience



def main():
    load_files()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    use_amp = CFG['amp'] and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    dataset = NeonTreeTiledDataset(
        rgb_dir        = "training/RGB",
        chm_dir        = "training/CHM",
        ann_dir        = "annotations/annotations",
        crop_size      = 512,
        stride         = 256,
        skip_empty     = True,
    )

    n_total = len(dataset)
    n_val   = int(n_total * CFG['val_fraction'])
    n_test  = int(n_total * CFG['test_fraction'])
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(CFG['seed'])
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    N_TRAIN_SUBSET = 100000
    if N_TRAIN_SUBSET and N_TRAIN_SUBSET < len(train_ds):
        indices = torch.randperm(len(train_ds), generator=generator)[:N_TRAIN_SUBSET]
        train_ds = torch.utils.data.Subset(train_ds, indices.tolist())
    
    N_DEV_SUBSET = 100000
    if N_DEV_SUBSET and N_DEV_SUBSET < len(val_ds):
        indices2 = torch.randperm(len(val_ds), generator=generator)[:N_DEV_SUBSET]
        val_ds = torch.utils.data.Subset(val_ds, indices2.tolist())
    

    # Save the test indices so test.py can reconstruct the exact same split
    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(test_ds.indices, ckpt_dir / 'test_indices.pt')

    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test} crops")

    train_loader = DataLoader(
        train_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = CFG['num_workers'],
        pin_memory  = torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = CFG['num_workers'],
        pin_memory  = torch.cuda.is_available(),
    )

    #! Model 
    model = build_4ch_faster_rcnn(CFG['num_classes']).to(device)
    print(f"\nModel: Faster R-CNN MobileNet")

    #Optimiser and scheduler for reducing LR on plateus
    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if 'backbone' not in n and p.requires_grad]

    optimiser = optim.SGD(
        [
            {'params': backbone_params, 'lr': CFG['lr'] * 0.1},   # slower for pretrained
            {'params': head_params,     'lr': CFG['lr']},
        ],
        momentum     = CFG['momentum'],
        weight_decay = CFG['weight_decay'],
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = 'min',
        patience = CFG['lr_patience'],
        factor   = CFG['lr_factor'],
        min_lr   = CFG['min_lr'],
    )

    early_stop = EarlyStopping(CFG['es_patience'])

    # ── Checkpointing setup ───────────────────────────────────────────────────
    log_path = ckpt_dir / 'training_log.csv'
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss',
                         'epoch_time_s'])

    best_val_loss  = float('inf')
    best_weights   = None

    # ── Epoch loop ────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  "
          f" {'Time':>7}")
    print(f"{'─'*65}")

    for p in model.backbone.parameters():
        p.requires_grad = False

    for epoch in range(1, CFG['num_epochs'] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimiser, device, scaler, CFG['grad_clip']
        )

        val_loss = evaluate_loss(model, val_loader, device)
        elapsed     = time.time() - t0

        # Step scheduler on val loss
        scheduler.step(val_loss)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = copy.deepcopy(model.state_dict())
            torch.save(
                {'epoch': epoch, 'val_loss': val_loss,
                 'model_state_dict': best_weights,
                 'optimiser_state_dict': optimiser.state_dict()},
                ckpt_dir / 'best_model.pth'
            )
            marker = '  ✓ best'
        else:
            marker = ''

        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
              f"{elapsed:>6.1f}s{marker}")

        # Append to log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss,f'{elapsed:.1f}'])

        # Early stopping check
        if early_stop.step(val_loss, epoch):
            print(f"\n  Early stopping at epoch {epoch}. "
                  f"Best val loss: {early_stop.best_loss:.4f} "
                  f"(epoch {early_stop.best_epoch})")
            break

    #Save final model
    torch.save(
        {'epoch': epoch, 'val_loss': val_loss,
         'model_state_dict': model.state_dict(),
         'optimiser_state_dict': optimiser.state_dict()},
        ckpt_dir / 'last_model.pth'
    )

    print(f"\nDone. Best val loss: {best_val_loss:.4f}  "
          f"(epoch {early_stop.best_epoch})")
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")


if __name__ == '__main__':
    main()