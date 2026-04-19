
import copy
import csv
import time
from pathlib import Path

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as T
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.tv_tensors import BoundingBoxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from load_display_data import load_files
from dataset import NeonTreeTiledDataset, collate_fn



CFG = dict(
    # Data split Split 80 / 10 / 10  
    val_fraction   = 0.10,
    test_fraction  = 0.10,
    seed           = 42,
    num_classes    = 2, 

    # Training
    num_epochs     = 50,
    batch_size     = 6,
    num_workers    = 6,           

    # Optimiser 
    lr             = 0.005,
    momentum       = 0.9,
    weight_decay   = 1e-4,

    # LR schedule
    lr_patience    = 4,
    lr_factor      = 0.5,         
    min_lr         = 1e-6,

    # Early stopping
    es_patience    = 5,          

    # Gradient clipping 
    grad_clip      = 5.0,
    
    checkpoint_dir = "checkpoints",
    amp            = True,        
)



def patch_first_conv(module):
    """ find and patch the first Conv2d to accept 4 channels."""
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
    Patching for 4 channel (RGB + CHM) input
    """
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    )

    patch_first_conv(model.backbone)

    #Replace classification head 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #Changing # of possible detections as there can be many trees in a tile
    model.rpn.pre_nms_top_n_train  = 4000   
    model.rpn.post_nms_top_n_train = 2000   
    model.rpn.pre_nms_top_n_test   = 2000
    model.rpn.post_nms_top_n_test  = 1000
    model.roi_heads.detections_per_img = 300

    #4-channel identity transform, must be redefined for 4dimensions 
    model.transform = GeneralizedRCNNTransform(
        min_size   = 512, 
        max_size   = 512, 
        image_mean = [0.0, 0.0, 0.0, 0.0],
        image_std  = [1.0, 1.0, 1.0, 1.0],
    )

    return model


#Target formatting 

def batch_to_targets(boxes_list, labels_list, device):
    """
    Convert collated batch lists into the list-of-dicts format faster R-CNN
    expects.  Labels are shifted +1 because 0 is reserved for background.
    """
    targets = []
    for boxes, labels in zip(boxes_list, labels_list):
        targets.append({
            'boxes':  boxes.to(device),
            'labels': (labels + 1).to(device),   # 0 = background
        })
    return targets


def train_one_epoch(model, loader, optimiser, device, scaler, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=90),         
        ])

    for batch in loader:
        images  = batch['image'].to(device)
        targets = batch_to_targets(batch['boxes'], batch['labels'], device)

        aug_images  = []
        aug_targets = []
        for img, tgt in zip(images, targets):
            boxes_tv = BoundingBoxes(
                tgt['boxes'].cpu(),
                format       = 'XYXY',
                canvas_size  = (img.shape[1], img.shape[2]),
            )
            img_aug, boxes_aug = train_transforms(img, boxes_tv)
            img_aug = img_aug.clamp(-3.0, 3.0)
            aug_images.append(img_aug.to(device))
            aug_targets.append({
                'boxes':  boxes_aug.data.to(device),
                'labels': tgt['labels'],
            })

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
    model.train()         
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
    print(f"Using device: {device}")

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


    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(test_ds.indices, ckpt_dir / 'test_indices.pt')

    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test} crops")

    #Data loaders for training and val

    train_loader = DataLoader(
        train_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = CFG['num_workers'],
        pin_memory  = torch.cuda.is_available(),
        persistent_workers = True,
        prefetch_factor = 2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = CFG['batch_size'],
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = CFG['num_workers'],
        pin_memory  = torch.cuda.is_available(),
        persistent_workers = True,
        prefetch_factor = 2,
    )

    #! Model 
    model = build_4ch_faster_rcnn(CFG['num_classes']).to(device)
    print(f"\nModel: Faster R-CNN MobileNet")

    #Optimiser and scheduler for reducing LR on plateus
    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if 'backbone' not in n and p.requires_grad]

    optimiser = optim.AdamW(
    [
        {'params': backbone_params, 'lr': 1e-5},  
        {'params': head_params,     'lr': 1e-4},  
    ],
    weight_decay = 1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = 'min',
        patience = CFG['lr_patience'],
        factor   = CFG['lr_factor'],
        min_lr   = CFG['min_lr'],
    )

    early_stop = EarlyStopping(CFG['es_patience'])

    #Checkpointing stuff

    best_val_loss  = float('inf')
    best_weights   = None

    #  Epoch loop
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Time':>7}")
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
            marker = 'best'
        else:
            marker = ''

        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  "
              f"{elapsed:>6.1f}s{marker}")

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