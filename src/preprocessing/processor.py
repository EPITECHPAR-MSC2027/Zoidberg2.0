import numpy as np
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

train_transform = A.Compose([
    # Resize to fixed size
    A.Resize(224, 224),

    # Geometric transformations
    A.Rotate(
        limit=10,  # Max 10°
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7
    ),

    A.ShiftScaleRotate(
        shift_limit=0.1,       # +10% translation
        scale_limit=0.1,       # 90-110% zoom
        rotate_limit=0,
        p=0.7,
        border_mode=cv2.BORDER_CONSTANT,
    ),

    # Horizontal flip only (lungs are symmetric)
    A.HorizontalFlip(p=0.5),

    # Elastic deformation
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.2
    ),

    # Perspective & distortion
    A.OneOf([
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
        A.OpticalDistortion(
            distort_limit=0.3,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
    ], p=0.3),

    # Intensity modifications
    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=1.0
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ], p=0.9),

    # Blur & sharpening
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MedianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=7, p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    ], p=0.4),

    # Noise
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.ISONoise(
            color_shift=(0.01, 0.03),
            intensity=(0.05, 0.3),
            p=1.0
        ),
        A.MultiplicativeNoise(
            multiplier=(0.95, 1.05),
            per_channel=True,
            p=1.0
        ),
    ], p=0.3),

    # Normalize & convert to tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
# Validation/Test
val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("\nAugmentation pipelines created!\n")

def dataset_to_arrays(dataset_split, image_size=(128, 128)):
    X = []
    y = []
    for example in dataset_split:
        # Convertir en RGB numpy
        img_array = np.array(example['image'].convert('RGB'))
        
        # Appliquer transform
        transformed = val_test_transform(image=img_array)
        img_tensor = transformed['image']
        
        # Convertir tenseur -> numpy -> flatten
        img_flattened = img_tensor.numpy().flatten()
        X.append(img_flattened)
        y.append(example['label'])
    
    return np.array(X), np.array(y)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
