#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic utilities
import os
import shutil
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
import cv2

# Skimage utilities
from skimage import data
from skimage.util import montage
from skimage.transform import rotate, resize

# Neural imaging
import nibabel as nib
import nilearn as nl
import nilearn.plotting as nlplt
# get_ipython().system('pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif')
import os
#os.system("pip install git+https://github.com/miykael/gif_your_nifti")

import gif_your_nifti.core as gif2nif  # Nifti to GIF converter

# Machine learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K, models, layers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate)
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger)
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# Settings
np.set_printoptions(precision=3, suppress=True)


# Segmentation class mapping
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'        # original label 4 remapped to 3
}

# Volume settings
VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE=128


# In[2]:


# Dataset paths
TRAIN_PATH = 'data/BRATS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALID_PATH = 'data/BRATS/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'

# Load sample images and mask
patient_id = 'BraTS20_Training_001'
modalities = ['flair', 't1', 't1ce', 't2', 'seg']

images = {mod: nib.load(f"{TRAIN_PATH}{patient_id}/{patient_id}_{mod}.nii").get_fdata() for mod in modalities}

# Visualize the center slices
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
slice_idx = images['flair'].shape[0] // 2 - 25
titles = ['FLAIR', 'T1', 'T1CE', 'T2', 'Mask']

for ax, mod, title in zip(axes, modalities, titles):
    ax.imshow(images[mod][:, :, slice_idx], cmap='gray' if mod != 'seg' else None)
    ax.set_title(f"Image {title}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# # Create model || U-Net: Convolutional Networks for Biomedical Image Segmentation
# he u-net is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. It has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin
# [more on](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
# ![official definiton](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
# 

# # Loss function
# **Dice coefficient**
# , which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap. The Dice coefficient was originally developed for binary data, and can be calculated as:
# 
# ![dice loss](https://wikimedia.org/api/rest_v1/media/math/render/svg/a80a97215e1afc0b222e604af1b2099dc9363d3b)
# 
# **As matrices**
# ![dice loss](https://www.jeremyjordan.me/content/images/2018/05/intersection-1.png)
# 
# [Implementation, (images above) and explanation can be found here](https://www.jeremyjordan.me/semantic-segmentation/)

# In[3]:


# # Dice Coefficient for multi-class (4 classes)
# def dice_coef(y_true, y_pred, smooth=1.0, class_num=4):
#     total_loss = 0
#     for i in range(class_num):
#         y_true_f = K.flatten(y_true[:, :, :, i])
#         y_pred_f = K.flatten(y_pred[:, :, :, i])
#         intersection = K.sum(y_true_f * y_pred_f)
#         dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#         total_loss += dice
#     return total_loss / class_num

# Per-Class Dice Coefficient
def dice_coef_class(y_true, y_pred, class_idx, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:, :, :, class_idx] * y_pred[:, :, :, class_idx]))
    denom = K.sum(K.square(y_true[:, :, :, class_idx])) + K.sum(K.square(y_pred[:, :, :, class_idx])) + epsilon
    return (2. * intersection) / denom

# def dice_coef_necrotic(y_true, y_pred):
#     return dice_coef_class(y_true, y_pred, class_idx=1)

# def dice_coef_edema(y_true, y_pred):
#     return dice_coef_class(y_true, y_pred, class_idx=2)

# def dice_coef_enhancing(y_true, y_pred):
#     return dice_coef_class(y_true, y_pred, class_idx=3)



# # Precision
# def precision(y_true, y_pred):
#     tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     return tp / (pp + K.epsilon())

# # Sensitivity (Recall)
# def sensitivity(y_true, y_pred):
#     tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     return tp / (possible_positives + K.epsilon())

# # Specificity
# def specificity(y_true, y_pred):
#     tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
#     possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
#     return tn / (possible_negatives + K.epsilon())


@tf.function
def dice_coef_necrotic(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, class_idx=1)

@tf.function
def dice_coef_edema(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, class_idx=2)

@tf.function
def dice_coef_enhancing(y_true, y_pred):
    return dice_coef_class(y_true, y_pred, class_idx=3)

@tf.function
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    total_loss = 0
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        total_loss += dice
    return total_loss / class_num

@tf.function
def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pp + K.epsilon())

@tf.function
def sensitivity(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (possible_positives + K.epsilon())

@tf.function
def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return tn / (possible_negatives + K.epsilon())


# In[4]:


# Source: https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a

def conv_block(x, filters, kernel_init, dropout_rate=None):
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(x)
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def build_unet(input_layer, kernel_init='he_normal', dropout_rate=0.2):
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

    outputs = Conv2D(4, (1, 1), activation='softmax')(c9)

    return Model(inputs=input_layer, outputs=outputs)

# Instantiate and compile the model
input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
model = build_unet(input_layer, kernel_init='he_normal', dropout_rate=0.2)

model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
    'accuracy',
    tf.keras.metrics.MeanIoU(num_classes=4),
    dice_coef, 
    precision, 
    sensitivity, 
    specificity, 
    dice_coef_necrotic, 
    dice_coef_edema,
    dice_coef_enhancing
    ]
)




# In[5]:


# # Visualize the model architecture
# plot_model(
#     model,
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir='TB',
#     dpi=70
# )


# In[6]:


# List all training/validation directories
train_val_dirs = [f.path for f in os.scandir(TRAIN_PATH) if f.is_dir()]

# Remove bad case
BAD_CASE = 'BraTS20_Training_355'
train_val_dirs = [d for d in train_val_dirs if not d.endswith(BAD_CASE)]

# Extract patient IDs from paths
def extract_ids(paths):
    return [os.path.basename(p) for p in paths]

train_val_ids = extract_ids(train_val_dirs)

# Train/Validation/Test split
train_test_ids, val_ids = train_test_split(train_val_ids, test_size=0.2, random_state=42)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15, random_state=42)


# **Override Keras sequence DataGenerator class**

# In[7]:


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(batch_ids)
        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        """Generates data containing batch_size samples."""
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240), dtype=np.uint8)

        for idx, patient_id in enumerate(batch_ids):
            case_path = os.path.join(TRAIN_PATH, patient_id)

            flair = nib.load(os.path.join(case_path, f'{patient_id}_flair.nii')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{patient_id}_t1ce.nii')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{patient_id}_seg.nii')).get_fdata()

            for slice_idx in range(VOLUME_SLICES):
                slice_num = slice_idx + VOLUME_START_AT
                X[slice_idx + VOLUME_SLICES * idx, :, :, 0] = cv2.resize(flair[:, :, slice_num], self.dim)
                X[slice_idx + VOLUME_SLICES * idx, :, :, 1] = cv2.resize(ce[:, :, slice_num], self.dim)
                y[slice_idx + VOLUME_SLICES * idx] = seg[:, :, slice_num]

        # Preprocess masks
        y[y == 4] = 3  # Convert label 4 to 3
        Y = tf.one_hot(y, depth=4)
        Y = tf.image.resize(Y, self.dim)

        # Normalize inputs
        X /= np.max(X)

        return X, Y

# Instantiate the generators
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)


# **Number of data used**
# for training / testing / validation

# In[8]:


# Callback: CSV Logger
csv_logger = CSVLogger('training.log', separator=',', append=False)

# Callback: Reduce LR on Plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# Callback: Model Checkpoint (optional, recommended)
checkpoint = ModelCheckpoint(
    filepath='data/UNET/model_per_class.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Combine callbacks
callbacks = [reduce_lr, csv_logger, checkpoint]


# In[9]:


# Clear any previous Keras session
K.clear_session()

# Train the model
history = model.fit(
    training_generator,
    validation_data=valid_generator,
    epochs=1,  # You can set to 35 when needed
    steps_per_epoch=len(train_ids),
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save("kaggle/working/3D_MRI_Brain_Tumor_Segmentation.h5")


# **Visualize the training process**

# In[10]:


import tensorflow as tf
import numpy as np
import nibabel as nib
import cv2

# Load the trained model with custom metrics
model = tf.keras.models.load_model(
    # "/kaggle/working/3D_MRI_Brain_Tumor_Segmentation.h5", 
    # "/kaggle/input/model-x80-dcs65/model_x81_dcs65.h5",
    "data/UNET/model_per_class.h5",
    custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'dice_coef_necrotic': dice_coef_necrotic,
        'dice_coef_edema': dice_coef_edema,
        'dice_coef_enhancing': dice_coef_enhancing
    }
)

def preprocess_image(image_path, slice_index):
    """Load, resize, and normalize a single slice from a NIfTI file."""
    img = nib.load(image_path).get_fdata()
    img_slice = img[:, :, slice_index]
    img_resized = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / np.max(img_resized)
    return img_resized

def predict_segmentation(flair_path, t1ce_path, slice_index):
    """Prepare input from FLAIR and T1CE images and predict segmentation."""
    X = np.zeros((1, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)
    X[0, :, :, 0] = preprocess_image(flair_path, slice_index)
    X[0, :, :, 1] = preprocess_image(t1ce_path, slice_index)
    
    prediction = model.predict(X)
    predicted_mask = np.argmax(prediction[0], axis=-1)
    return predicted_mask

# Example usage
# flair_path = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_010/BraTS20_Validation_010_flair.nii'
# t1ce_path = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_010/BraTS20_Validation_010_t1ce.nii'

flair_path = 'data/BRATS/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_010/BraTS20_Validation_010_flair.nii'
t1ce_path = 'data/BRATS/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_010/BraTS20_Validation_010_t1ce.nii'
slice_idx = 75

predicted_mask = predict_segmentation(flair_path, t1ce_path, slice_idx)

# Directly visualize the predicted mask
# visualize_prediction(predicted_mask)



# In[11]:


import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm  # progress bar

# Settings
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22

def preprocess_volume(flair_path, t1ce_path):
    """Load, resize, and normalize an entire 3D volume for a patient."""
    flair = nib.load(flair_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()

    volume = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    for i in range(VOLUME_SLICES):
        slice_idx = VOLUME_START_AT + i
        volume[i, :, :, 0] = cv2.resize(flair[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)) / np.max(flair)
        volume[i, :, :, 1] = cv2.resize(t1ce[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)) / np.max(t1ce)

    return volume

def predict_volume(model, flair_path, t1ce_path):
    """Predict the full volume slice-by-slice."""
    X_volume = preprocess_volume(flair_path, t1ce_path)  # shape (100, 128, 128, 2)
    
    predicted_volume = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    for i in range(VOLUME_SLICES):
        X_slice = np.expand_dims(X_volume[i], axis=0)  # shape (1, 128, 128, 2)
        prediction = model.predict(X_slice, verbose=0)
        predicted_mask = np.argmax(prediction[0], axis=-1)
        predicted_volume[i] = predicted_mask

    return predicted_volume


def process_validation_or_test_set(model, patient_ids, base_path):
    """Predict segmentations for a list of patients."""
    all_predictions = {}
    
    for pid in tqdm(patient_ids, desc='Predicting Patients'):
        flair_path = os.path.join(base_path, pid, f'{pid}_flair.nii')
        t1ce_path = os.path.join(base_path, pid, f'{pid}_t1ce.nii')

        predicted_volume = predict_volume(model, flair_path, t1ce_path)
        all_predictions[pid] = predicted_volume

    return all_predictions


# In[12]:


# Paths
# VALIDATION_DATASET_PATH = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'
# TRAIN_DATASET_PATH = '/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = "data/BRATS/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/"
TRAIN_DATASET_PATH = "data/BRATS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"

# Example: Validation patients
# List only folders (patients) — ignore CSV or other files
validation_patient_ids = [d for d in os.listdir(VALIDATION_DATASET_PATH) 
                          if os.path.isdir(os.path.join(VALIDATION_DATASET_PATH, d))]

# Example: Testing patients (if you split already)
# testing_patient_ids = your_test_split_ids_list

# Predict Validation Set
validation_predictions = process_validation_or_test_set(model, validation_patient_ids, VALIDATION_DATASET_PATH)

# Predict Test Set (if you have)
# test_predictions = process_validation_or_test_set(model, testing_patient_ids, TRAIN_DATASET_PATH)


# #SAM#

# In[13]:


import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# Settings
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
OUTPUT_MASKS_FOLDER = 'kaggle/working/pseudo_labels/'  # Path where you save predicted masks
model = tf.keras.models.load_model(
    # '/kaggle/input/model-x80-dcs65/model_x81_dcs65.h5',
    'data/UNET/model_per_class.h5',
    custom_objects={
        'dice_coef': dice_coef,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'dice_coef_necrotic': dice_coef_necrotic,
        'dice_coef_edema': dice_coef_edema,
        'dice_coef_enhancing': dice_coef_enhancing
    }
)


os.makedirs(OUTPUT_MASKS_FOLDER, exist_ok=True)

def preprocess_volume(flair_path, t1ce_path):
    flair = nib.load(flair_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    volume = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2), dtype=np.float32)

    for i in range(VOLUME_SLICES):
        slice_idx = VOLUME_START_AT + i
        volume[i, :, :, 0] = cv2.resize(flair[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)) / np.max(flair)
        volume[i, :, :, 1] = cv2.resize(t1ce[:, :, slice_idx], (IMG_SIZE, IMG_SIZE)) / np.max(t1ce)

    return volume

# New OUTPUT FOLDER
OUTPUT_DATASET_FOLDER = 'kaggle/working/dataset/'  # Not pseudo_labels/
os.makedirs(os.path.join(OUTPUT_DATASET_FOLDER, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DATASET_FOLDER, "masks"), exist_ok=True)

def predict_and_save_pseudolabels(model, patient_ids, base_path):
    for pid in tqdm(patient_ids, desc='Generating pseudo-labels'):
        flair_path = os.path.join(base_path, pid, f'{pid}_flair.nii')
        t1ce_path = os.path.join(base_path, pid, f'{pid}_t1ce.nii')

        X_volume = preprocess_volume(flair_path, t1ce_path)

        for i in range(VOLUME_SLICES):
            X_slice = np.expand_dims(X_volume[i], axis=0)  # (1, 128, 128, 2)
            prediction = model.predict(X_slice, verbose=0)
            predicted_mask = np.argmax(prediction[0], axis=-1)  # (128,128)

            # Save input images (choose one channel, e.g., FLAIR)
            image_save_path = os.path.join(OUTPUT_DATASET_FOLDER, "images", f'{pid}_slice_{i+VOLUME_START_AT}.png')
            flair_image = (X_volume[i, :, :, 0] * 255).astype(np.uint8)  # 0th channel is FLAIR
            cv2.imwrite(image_save_path, flair_image)

            # Save predicted mask
            mask_save_path = os.path.join(OUTPUT_DATASET_FOLDER, "masks", f'{pid}_slice_{i+VOLUME_START_AT}.png')
            cv2.imwrite(mask_save_path, (predicted_mask * 85).astype(np.uint8))  # Scale mask for visibility



# Set path
TRAIN_DATASET_PATH = 'data/BRATS/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

# Get patient IDs
# List only folders (patients), not any unwanted files
train_patient_ids = [d for d in os.listdir(TRAIN_DATASET_PATH) 
                     if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, d))]

# Select only the first 10 patients
train_patient_ids = train_patient_ids[:10]


# Predict and save
predict_and_save_pseudolabels(model, train_patient_ids, TRAIN_DATASET_PATH)



# In[17]:


import gc
import torch
K.clear_session()

# 1. Manually delete all model variables (if you know them)
try:
    del model
except:
    pass

try:
    del sam
except:
    pass

# 2. Empty all existing tensors from CUDA
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

print("✅ All models and GPU memory cleaned!")


# In[18]:


#get_ipython().system('pip install git+https://github.com/facebookresearch/segment-anything.git')
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from segment_anything import sam_model_registry


MODEL_TYPE = "vit_b"  # Options: 'vit_b', 'vit_l', 'vit_h'
# SAM_CKPT_PATH = "/kaggle/input/segment-anything/pytorch/vit-b/1/model.pth"  # Pretrained SAM checkpoint path
SAM_CKPT_PATH = "data/segment-anything-pytorch-vit-b-v1/model.pth"
DATASET_PATH = "kaggle/working/dataset"  # Pseudo-labeled dataset: images/ and masks/
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
sam.to(DEVICE)

sam_encoder = sam.image_encoder
sam_encoder.eval()
for param in sam_encoder.parameters():
    param.requires_grad = False


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

        image = cv2.imread(img_path)  # BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # Binary mask

        if self.transform:
            image = self.transform(image)  # Already returns tensor (C,H,W)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

        mask = torch.tensor(mask).unsqueeze(0)  # (1, H, W)

        return image, mask


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])


train_dataset = SegmentationDataset(
    images_dir=os.path.join(DATASET_PATH, "images"),
    masks_dir=os.path.join(DATASET_PATH, "masks"),
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# In[ ]:


class SimpleSAMDecoder(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super(SimpleSAMDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)  # Binary output
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)  # Freeze encoder
        output = self.decoder(features)
        return output


# In[ ]:


model = SimpleSAMDecoder(sam_encoder).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[ ]:


def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()
    
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)  # (B, 1, H, W)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Metrics
        running_dice += dice_coefficient(torch.sigmoid(outputs), targets).item()
        running_iou += iou_score(torch.sigmoid(outputs), targets).item()
        running_acc += pixel_accuracy(torch.sigmoid(outputs), targets).item()

    # Average over all batches
    avg_loss = running_loss / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    avg_acc = running_acc / len(train_loader)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] -> Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Pixel Acc: {avg_acc:.4f}")


# In[ ]:




