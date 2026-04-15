from load_display_data import load_files
from dataset import NeonTreeTiledDataset, collate_fn 
from torch.utils.data import DataLoader

def main():
    print("Hello from bip-proj!")
    load_files()  # downloads + unzips everything first
    from dataset import NeonTreeTiledDataset, collate_fn   # was NeonTreeDataset

    #Training dataset
    dataset = NeonTreeTiledDataset(
        rgb_dir="training/RGB",
        chm_dir="training/CHM",
        ann_dir="annotations/annotations",
        crop_size=512,    # px per crop
        stride=256,       # 50% overlap
        skip_empty=True,  # discard crops with no trees
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)} samples")

    # Quick check — print the first batch
    batch = next(iter(loader))
    print("Image batch shape:", batch['image'].shape)  # (4, 4, 512, 512)
    print("Boxes per image:  ", [b.shape for b in batch['boxes']])

if __name__ == "__main__":
    main()
