#!/usr/bin/env python
# coding: utf-8

"""
Brain Tumor Segmentation Pipeline
- Uses U-Net for initial segmentation
- Uses Segment Anything (SAM) for further refinement
- TensorFlow + PyTorch hybrid training
"""

# === Imports ===
import os
import gc
import cv2
import glob
import torch
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageOps
from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from tensorflow.keras.models import Model

# Optional: For plotting NIfTI
import nilearn.plotting as nlplt
import gif_your_nifti.core as gif2nif

import random
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# === Global Settings ===
np.set_printoptions(precision=3, suppress=True)

# === Constants ===
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
TRAIN_PATH = 'data/BRATS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALID_PATH = 'data/BRATS/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

# === GPU Memory Cleanup ===
def clear_gpu_memory(models_to_clear=None, variables_to_clear=None):
    print("\n🔁 Clearing GPU memory...")
    try:
        K.clear_session()
        print("✅ Keras session cleared.")
    except Exception as e:
        print(f"⚠️ Could not clear Keras session: {e}")
    if models_to_clear:
        for model in models_to_clear:
            try:
                del model
            except Exception as e:
                print(f"⚠️ Could not delete model: {e}")
    if variables_to_clear:
        for var in variables_to_clear:
            try:
                del var
            except Exception as e:
                print(f"⚠️ Could not delete variable: {e}")
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✅ PyTorch cache cleared.")
    except Exception as e:
        print(f"⚠️ PyTorch cleanup failed: {e}")
    print("🧹 GPU memory cleared successfully!\n")

# === Load Example Patient for Visualization ===
def load_and_visualize_sample(patient_id='BraTS20_Training_001'):
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']
    images = {
        mod: nib.load(f"{TRAIN_PATH}{patient_id}/{patient_id}_{mod}.nii").get_fdata()
        for mod in modalities
    }
    slice_idx = images['flair'].shape[0] // 2 - 25
    titles = ['FLAIR', 'T1', 'T1CE', 'T2', 'Mask']
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    for ax, mod, title in zip(axes, modalities, titles):
        ax.imshow(images[mod][:, :, slice_idx], cmap='gray' if mod != 'seg' else None)
        ax.set_title(f"Image {title}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()



# === U-Net Model Definition ===
def conv_block(x, filters, kernel_init, dropout_rate=None):
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(x)
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def build_unet_model(img_size=IMG_SIZE, channels=2, num_classes=4, dropout_rate=0.2, learning_rate=1e-3):
    input_layer = Input((img_size, img_size, channels))
    kernel_init = 'he_normal'

    # Encoder
    c1 = conv_block(input_layer, 32, kernel_init)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 64, kernel_init)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 128, kernel_init)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 256, kernel_init)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = conv_block(p4, 512, kernel_init, dropout_rate)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=kernel_init)(u6)
    u6 = concatenate([c4, u6])
    c6 = conv_block(u6, 256, kernel_init)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=kernel_init)(u7)
    u7 = concatenate([c3, u7])
    c7 = conv_block(u7, 128, kernel_init)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=kernel_init)(u8)
    u8 = concatenate([c2, u8])
    c8 = conv_block(u8, 64, kernel_init)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=kernel_init)(u9)
    u9 = concatenate([c1, u9])
    c9 = conv_block(u9, 32, kernel_init)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=input_layer, outputs=outputs)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=num_classes)]
    )
    return model

# === Data Generator for Training ===
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras model training."""
    def __init__(self, list_IDs, data_path, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        # super().__init__(**kwargs)
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240), dtype=np.uint8)

        for idx, patient_id in enumerate(batch_ids):
            case_path = os.path.join(self.data_path, patient_id)
            flair = nib.load(os.path.join(case_path, f'{patient_id}_flair.nii')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{patient_id}_t1ce.nii')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{patient_id}_seg.nii')).get_fdata()

            for slice_idx in range(VOLUME_SLICES):
                slice_num = slice_idx + VOLUME_START_AT
                X[slice_idx + VOLUME_SLICES * idx, :, :, 0] = cv2.resize(flair[:, :, slice_num], self.dim)
                X[slice_idx + VOLUME_SLICES * idx, :, :, 1] = cv2.resize(ce[:, :, slice_num], self.dim)
                y[slice_idx + VOLUME_SLICES * idx] = seg[:, :, slice_num]

        y[y == 4] = 3
        Y = tf.one_hot(y, depth=4)
        Y = tf.image.resize(Y, self.dim)

        X /= np.max(X)

        return X, Y

# === Training Utilities ===
def compile_callbacks(log_path='training.log', checkpoint_path='data/UNET/model_per_class.weights.h5'):
    csv_logger = CSVLogger(log_path, separator=',', append=False)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    return [reduce_lr, csv_logger, checkpoint]


def train_model(model, train_gen, val_gen, epochs=1, steps_per_epoch=None, callbacks=None):
    print("\n🚀 Starting U-Net training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )
    print("\n✅ Training complete.")
    return history


def save_model(model, path="data/working/3D_MRI_Brain_Tumor_Segmentation.h5"):
    model.save(path)
    print(f"💾 Model saved to: {path}")

# === Volume Prediction Utilities ===
def preprocess_volume(flair_path, t1ce_path, img_size=IMG_SIZE):
    flair = nib.load(flair_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    volume = np.zeros((VOLUME_SLICES, img_size, img_size, 2), dtype=np.float32)
    for i in range(VOLUME_SLICES):
        slice_idx = VOLUME_START_AT + i
        volume[i, :, :, 0] = cv2.resize(flair[:, :, slice_idx], (img_size, img_size)) / np.max(flair)
        volume[i, :, :, 1] = cv2.resize(t1ce[:, :, slice_idx], (img_size, img_size)) / np.max(t1ce)
    return volume


def predict_volume(model, flair_path, t1ce_path):
    X_volume = preprocess_volume(flair_path, t1ce_path)
    predicted_volume = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(VOLUME_SLICES):
        X_slice = np.expand_dims(X_volume[i], axis=0)
        prediction = model.predict(X_slice, verbose=0)
        predicted_mask = np.argmax(prediction[0], axis=-1)
        predicted_volume[i] = predicted_mask
    return predicted_volume


def process_validation_or_test_set(model, patient_ids, base_path):
    all_predictions = {}
    for pid in tqdm(patient_ids, desc='Predicting Patients'):
        flair_path = os.path.join(base_path, pid, f'{pid}_flair.nii')
        t1ce_path = os.path.join(base_path, pid, f'{pid}_t1ce.nii')
        predicted_volume = predict_volume(model, flair_path, t1ce_path)
        all_predictions[pid] = predicted_volume
    return all_predictions


# === SAM Training Utilities ===
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = torch.tensor(mask).unsqueeze(0)
        return image, mask


def get_sam_dataloader(images_dir, masks_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    dataset = SegmentationDataset(images_dir, masks_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class SimpleSAMDecoder(nn.Module):
    def __init__(self, encoder, num_classes=1, output_size=(128, 128)):
        super(SimpleSAMDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        self.output_size = output_size

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        x = self.decoder(features)
        x = nn.functional.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x



def train_sam_model(model, dataloader, num_epochs, device, lr=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):
        preds = (preds > threshold).float()
        targets = (targets > threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return ((2. * intersection + eps) / (union + eps)).mean()

    def iou_score(preds, targets, threshold=0.5, eps=1e-6):
        preds = (preds > threshold).float()
        targets = (targets > threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        return ((intersection + eps) / (union + eps)).mean()

    def pixel_accuracy(preds, targets, threshold=0.5):
        preds = (preds > threshold).float()
        correct = (preds == targets).float()
        return correct.sum() / correct.numel()

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_dice, running_iou, running_acc = 0, 0, 0, 0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(torch.sigmoid(outputs), targets).item()
            running_iou += iou_score(torch.sigmoid(outputs), targets).item()
            running_acc += pixel_accuracy(torch.sigmoid(outputs), targets).item()

        n = len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {running_loss/n:.4f}, Dice: {running_dice/n:.4f}, IoU: {running_iou/n:.4f}, Pixel Acc: {running_acc/n:.4f}")




# === Main Function ===
def main(mode="hpc"):
    print("🧠 Brain Tumor Segmentation Pipeline Starting...")

    # if mode == "local":
    #     print("🔬 Running in LOCAL mode: Using subset of data for quick verification.")
    #     train_ids_local = os.listdir(TRAIN_PATH)[:2]
    #     val_ids_local = os.listdir(TRAIN_PATH)[-1:]
    #     epochs = 1
    #     steps_per_epoch = len(train_ids_local)
    # else:
    #     print("🚀 Running in HPC mode: Using full dataset.")
    #     train_ids_local = os.listdir(TRAIN_PATH)
    #     val_ids_local = os.listdir(TRAIN_PATH)[:5]
    #     epochs = 35
    #     steps_per_epoch = len(train_ids_local)
    def is_valid_patient_dir(path, name_prefix="BraTS20"):
        full_path = os.path.join(TRAIN_PATH, path)
        return os.path.isdir(full_path) and path.startswith(name_prefix)

    all_patient_dirs = [d for d in os.listdir(TRAIN_PATH) if is_valid_patient_dir(d)]

    if mode == "local":
        print("🔬 Running in LOCAL mode: Using subset of data for quick verification.")
        train_ids_local = all_patient_dirs[:2]
        val_ids_local = all_patient_dirs[-1:]
        epochs = 1
        steps_per_epoch = len(train_ids_local)
    else:
        print("🚀 Running in HPC mode: Using full dataset.")
        train_ids_local = all_patient_dirs
        val_ids_local = all_patient_dirs[:5]
        epochs = 35
        steps_per_epoch = len(train_ids_local)


    # Phase 1: Train U-Net
    train_gen = DataGenerator(train_ids_local, TRAIN_PATH)
    val_gen = DataGenerator(val_ids_local, TRAIN_PATH)

    model = build_unet_model()
    callbacks = compile_callbacks()
    history = train_model(model, train_gen, val_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    save_model(model)

    # Phase 2: Generate Pseudo Labels
    def predict_and_save_pseudolabels(model, patient_ids, base_path):
        save_img_dir = "data/working/dataset/images"
        save_mask_dir = "data/working/dataset/masks"
        os.makedirs(save_img_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)

        for pid in tqdm(patient_ids, desc="Generating Pseudo Labels"):
            flair_path = os.path.join(base_path, pid, f"{pid}_flair.nii")
            t1ce_path = os.path.join(base_path, pid, f"{pid}_t1ce.nii")
            volume = preprocess_volume(flair_path, t1ce_path)
            for i in range(VOLUME_SLICES):
                X_slice = np.expand_dims(volume[i], axis=0)
                pred = model.predict(X_slice, verbose=0)
                mask = np.argmax(pred[0], axis=-1).astype(np.uint8)
                img = (volume[i][:, :, 0] * 255).astype(np.uint8)

                Image.fromarray(img).save(os.path.join(save_img_dir, f"{pid}_{i:03d}.png"))
                Image.fromarray(mask * 85).save(os.path.join(save_mask_dir, f"{pid}_{i:03d}.png"))  # Class to grayscale

    pseudo_ids = train_ids_local[:1] if mode == "local" else train_ids_local
    predict_and_save_pseudolabels(model, pseudo_ids, TRAIN_PATH)

    # Phase 3: Fine-tune SAM
    from segment_anything import sam_model_registry  # Ensure this is installed & accessible
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    MODEL_TYPE = "vit_b"
    SAM_CKPT_PATH = "data/segment-anything-pytorch-vit-b-v1/model.pth"
    DATASET_PATH = "data/working/dataset"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
    sam.to(DEVICE)
    sam_encoder = sam.image_encoder
    sam_encoder.eval()
    for param in sam_encoder.parameters():
        param.requires_grad = False

    sam_loader = get_sam_dataloader(
        images_dir=os.path.join(DATASET_PATH, "images"),
        masks_dir=os.path.join(DATASET_PATH, "masks"),
        batch_size=2 if mode == "local" else 8
    )

    sam_model = SimpleSAMDecoder(sam_encoder).to(DEVICE)
    train_sam_model(sam_model, sam_loader, num_epochs=1 if mode == "local" else 10, device=DEVICE)



if __name__ == "__main__":
    import sys
    mode = "local"
    main(mode)
    # main()




# code structure
# - U-Net model definition line 110
# - DataGenerator definition line 166
# - Training loop and utils line 216
# - Volume prediction line 257
# - SAM-based model training line 290