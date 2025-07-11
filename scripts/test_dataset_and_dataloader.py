
import os
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Step 1: List image and mask files ---
image_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_images"
mask_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_mask"

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

# Ensure filenames match
matched_image_paths = []
matched_mask_paths = []

for img_name in image_files:
    if img_name in mask_files:
        matched_image_paths.append(os.path.join(image_dir, img_name))
        matched_mask_paths.append(os.path.join(mask_dir, img_name))
    else:
        print(f"Warning: No matching mask for {img_name}")

print(f"Matched {len(matched_image_paths)} image-mask pairs.")

# --- Step 2: Define contrast stretch function ---
def stretch_contrast(img, low=2, high=98):
    vmin = np.percentile(img, low)
    vmax = np.percentile(img, high)
    stretched = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    return stretched

# --- Step 3: Define Dataset ---
class SARDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        print(f"Loading: {img_path} <--> {mask_path}")

        with rasterio.open(img_path) as img_src:
            image = img_src.read(1).astype(np.float32)

        with rasterio.open(mask_path) as mask_src:
            mask = mask_src.read(1).astype(np.float32)
            mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask

# --- Step 4: Apply transform and load with DataLoader ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

dataset = SARDataset(matched_image_paths, matched_mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# --- Step 5: Visualize up to 5 image+mask pairs with contrast stretch ---
num_samples = min(5, len(dataset))
fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(6, 1.8 * num_samples))

if num_samples == 1:
    axes = np.array(axes).reshape(1, 2)

for i in range(num_samples):
    img, mask = dataset[i]
    img_np = img.squeeze().numpy()
    mask_np = mask.squeeze().numpy()

    img_np_stretched = stretch_contrast(img_np)

    axes[i, 0].imshow(img_np_stretched, cmap="gray")
    axes[i, 0].set_title(f"Image {i+1}", fontsize=9)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask_np, cmap="gray")
    axes[i, 1].set_title(f"Mask {i+1}", fontsize=9)
    axes[i, 1].axis("off")

plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.tight_layout()
plt.show()
