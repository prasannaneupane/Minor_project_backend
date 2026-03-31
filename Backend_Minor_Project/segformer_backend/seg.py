import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import cv2
import matplotlib.pyplot as plt

import os

from torch.utils.data import Dataset


class PlantDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, augment=None, img_size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.augment = augment
        self.img_size = img_size

        # ---------------------------
        # Match image-mask pairs safely
        # ---------------------------
        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))

        img_stems = {os.path.splitext(f)[0]: f for f in img_files}
        mask_stems = {os.path.splitext(f)[0]: f for f in mask_files}

        common = sorted(set(img_stems.keys()) & set(mask_stems.keys()))

        self.images = [img_stems[k] for k in common]
        self.masks = [mask_stems[k] for k in common]

        # ---------------------------
        # Helpful dataset report
        # ---------------------------
        print("Total images:", len(img_files))
        print("Total masks :", len(mask_files))
        print("Matched pairs:", len(common))
        print("Dropped images without masks:", len(set(img_stems) - set(mask_stems)))
        print("Dropped masks without images:", len(set(mask_stems) - set(img_stems)))

        if len(common) == 0:
            raise RuntimeError("No matching image-mask pairs found!")

    # ---------------------------
    # Dataset length
    # ---------------------------
    def __len__(self):
        return len(self.images)

    # ---------------------------
    # Load sample
    # ---------------------------
    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # ---------------------------
        # Force SAME size (critical)
        # ---------------------------
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,  # preserves labels
        )

        # Binary mask (0 / 1)
        mask = (mask > 0).astype(np.uint8)

        # ---------------------------
        # Augmentation
        # ---------------------------
        if self.augment:
            augmented = self.augment(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # ---------------------------
        # SegFormer processor
        # ---------------------------
        encoded = self.processor(
            images=img, segmentation_maps=mask, return_tensors="pt"
        )

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = encoded["labels"].long()

        return encoded
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_one_hot = (
            torch.nn.functional.one_hot(targets, num_classes=probs.shape[1])
            .permute(0, 3, 1, 2)
            .float()
        )

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class SegformerLightning(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-5, freeze_epochs=3):

        super().__init__()
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b1-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # ===============================
        # NEW: decoder dropout (regularization)
        # ===============================
        self.model.decode_head.dropout = nn.Dropout2d(0.3)

        # ===============================
        # UPDATED: CE loss with ignore index
        # ===============================
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=255)

        # ===============================
        # NEW: Dice loss
        # ===============================
        self.dice = DiceLoss()

        # metrics
        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)

        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self.freeze_epochs = freeze_epochs

        # freeze encoder initially
        for p in self.model.segformer.encoder.parameters():
            p.requires_grad = False

    # ---------------- FORWARD ----------------
    def forward(self, **batch):
        return self.model(**batch)

    # ---------------- LOSS FUNCTION ----------------
    def compute_loss(self, logits, labels):
        ce_loss = self.ce(logits, labels)
        dice_loss = self.dice(logits, labels)
        return 0.5 * ce_loss + 0.5 * dice_loss

    # ---------------- TRAIN ----------------
    def training_step(self, batch, batch_idx):

        out = self(**batch)
        logits = out.logits
        labels = batch["labels"]

        logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = self.compute_loss(logits, labels)
        preds = logits.argmax(1)

        self.train_iou.update(preds, labels)
        self.train_acc.update(preds, labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_mIoU", self.train_iou.compute(), prog_bar=True)
        self.log("train_accuracy", self.train_acc.compute(), prog_bar=True)

        self.train_iou.reset()
        self.train_acc.reset()

    # ---------------- VALID ----------------
    def validation_step(self, batch, batch_idx):

        out = self(**batch)
        logits = out.logits
        labels = batch["labels"]

        logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = self.compute_loss(logits, labels)
        preds = logits.argmax(1)

        self.val_iou.update(preds, labels)
        self.val_acc.update(preds, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_mIoU", self.val_iou.compute(), prog_bar=True)
        self.log("val_accuracy", self.val_acc.compute(), prog_bar=True)

        self.val_iou.reset()
        self.val_acc.reset()

    # ---------------- TEST ----------------
    def test_step(self, batch, batch_idx):

        out = self(**batch)
        logits = out.logits
        labels = batch["labels"]

        logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = self.compute_loss(logits, labels)
        preds = logits.argmax(1)

        self.test_iou.update(preds, labels)
        self.test_acc.update(preds, labels)

        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        self.log("test_mIoU", self.test_iou.compute())
        self.log("test_accuracy", self.test_acc.compute())

        self.test_iou.reset()
        self.test_acc.reset()

    # ---------------- UNFREEZE BACKBONE ----------------
    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_epochs:
            print("Unfreezing encoder...")
            for p in self.model.segformer.encoder.parameters():
                p.requires_grad = True

    # ---------------- OPTIMIZER ----------------
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-3
        )

        # updated scheduler scale
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return [optimizer], [scheduler]
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor

# ---------------- LOAD MODEL ----------------
model = SegformerLightning.load_from_checkpoint(
    "/mnt/Data/Plant_Disease_Detection/Split(0.7,0.2,0.1), LR (1e-5), WD(1e-3) 20 epoc NL/best.ckpt"
)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------- LOAD PROCESSOR ----------------
processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b1-finetuned-ade-512-512"
)

# ---------------- DISPLAY AND OVERLAY ----------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ---------------- LOAD IMAGE ----------------
image_path = "/mnt/Data/Plant_Disease_Detection/apple_black_rot_google_0221.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------- MODEL INPUT ----------------
inputs = processor(images=image_rgb, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# ---------------- MODEL PREDICTION ----------------
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    logits = F.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode="bilinear",
        align_corners=False
    )
    preds = torch.argmax(logits, dim=1).cpu().numpy()[0]  # H x W, 0=healthy, 1=disease

# ---------------- CREATE BINARY MASK ----------------
mask = (preds == 1).astype(np.uint8)

# ---------------- COLORED MASK ----------------
disease_mask_colored = np.zeros_like(image_rgb)
disease_mask_colored[preds == 1] = [255, 0, 0]  # Red color for disease

# ---------------- OVERLAY ----------------
alpha = 0.5
# Pixel-wise overlay
overlay = image_rgb.copy()
overlay[preds == 1] = (
    (1 - alpha) * image_rgb[preds == 1] + alpha * disease_mask_colored[preds == 1]
).astype(np.uint8)

# ---------------- DISPLAY ----------------
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay - Disease Highlighted")
plt.imshow(overlay)
plt.axis("off")

plt.show()

# ---------------- CALCULATE SEVERITY ----------------
def get_leaf_mask(image):
    """Simple green thresholding to get leaf region"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
    return leaf_mask

def calculate_severity(leaf_mask, disease_mask):
    """Compute severity %"""
    leaf_pixels = np.sum(leaf_mask > 0)
    disease_pixels = np.sum(disease_mask == 1)
    if leaf_pixels == 0:
        return 0.0
    severity = (disease_pixels / leaf_pixels) * 100
    return min(severity, 100.0)  # clamp to 100%

# Leaf mask
leaf_mask = get_leaf_mask(image)

# Disease mask from model
disease_mask = preds

severity = calculate_severity(leaf_mask, disease_mask)
print(f"Disease Severity: {severity:.2f}%")