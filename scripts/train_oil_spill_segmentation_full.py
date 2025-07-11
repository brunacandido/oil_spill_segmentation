
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

# --- Paths ---
image_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_images"
mask_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/data/01_Train_Val_Oil_Spill_mask"
output_dir = "C:/Users/Bruna/Documents/repos/oil_spill_segmentation/outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Load file paths ---
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])

matched_image_paths = []
matched_mask_paths = []

for img_name in image_files:
    if img_name in mask_files:
        matched_image_paths.append(os.path.join(image_dir, img_name))
        matched_mask_paths.append(os.path.join(mask_dir, img_name))

print(f"Matched {len(matched_image_paths)} image-mask pairs.")

# --- Dataset definition ---
class SARDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as img_src:
            image = img_src.read(1).astype(np.float32)

        with rasterio.open(self.mask_paths[idx]) as mask_src:
            mask = mask_src.read(1).astype(np.float32)
            mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask

# --- Transformations ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

# --- Dataset and DataLoaders ---
full_dataset = SARDataset(matched_image_paths, matched_mask_paths, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# --- Model ---
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Loss and optimizer ---
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training ---
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation with metrics
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

            # Salvar predições como imagens
            for j in range(images.size(0)):
                pred_img = preds[j][0].cpu().numpy()
                mask_img = masks[j][0].cpu().numpy()
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                ax[0].imshow(pred_img, cmap="gray")
                ax[0].set_title("Prediction")
                ax[1].imshow(mask_img, cmap="gray")
                ax[1].set_title("Ground Truth")
                for a in ax: a.axis("off")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/epoch{epoch+1}_sample{i*images.size(0)+j}.png")
                plt.close()

    avg_val_loss = val_loss / len(val_loader)

    # Métricas
    preds_flat = np.concatenate(all_preds).reshape(-1)
    targets_flat = np.concatenate(all_targets).reshape(-1)
    iou = jaccard_score(targets_flat, preds_flat)
    precision = precision_score(targets_flat, preds_flat)
    recall = recall_score(targets_flat, preds_flat)
    f1 = f1_score(targets_flat, preds_flat)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    print(f"  IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# --- Salvar modelo final ---
torch.save(model.state_dict(), os.path.join(output_dir, "unet_oilspill.pth"))
print("Modelo salvo em:", os.path.join(output_dir, "unet_oilspill.pth"))
