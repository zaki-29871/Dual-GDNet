import numpy as np
import cv2
import os
import shutil

# Setting
original_data_folder = r'F:\Dataset\KITTI 2015\training'
destination_folder = r'F:\Dataset\KITTI 2015 Data Augmentation'
blur_kernel = 5
scale_ratio = 1.5
gaussian_noise_ratio = 0.2  # [0, 1]
gaussian_noise_mean = 0
gaussian_noise_std = 0.15  # [0, 1]

if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder, exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'image_2'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'image_3'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'disp_occ_0'), exist_ok=True)

# Copy original images
for i in range(0, 1):
    print(f'Copy index {i}')
    # All (376, 1241, 3) uint8
    X1_path = f'{i:06d}_10'
    X2_path = f'{i:06d}_10'
    Y_path = f'{i:06d}_10'

    X1 = cv2.imread(os.path.join(original_data_folder, 'image_2', f'{X1_path}.png'))
    X2 = cv2.imread(os.path.join(original_data_folder, 'image_3', f'{X2_path}.png'))
    Y = cv2.imread(os.path.join(original_data_folder, 'disp_occ_0', f'{Y_path}.png'))

    cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'), X1)
    cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'), X2)
    cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'), Y)

# Vertical flipping images
for X1_path in os.listdir(os.path.join(destination_folder, 'image_2')):
    print(f'Vertical flipping {X1_path}')
    X1_path = X1_path.split('.')[0]
    X2_path = X1_path.split('.')[0]
    Y_path = X1_path.split('.')[0]

    X1 = cv2.imread(os.path.join(original_data_folder, 'image_2', f'{X1_path}.png'))
    X2 = cv2.imread(os.path.join(original_data_folder, 'image_3', f'{X2_path}.png'))
    Y = cv2.imread(os.path.join(original_data_folder, 'disp_occ_0', f'{Y_path}.png'))

    # 0 is vertical flip
    X1 = cv2.flip(X1, 0)
    X2 = cv2.flip(X2, 0)
    Y = cv2.flip(Y, 0)

    cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_flip.png'), X1)
    cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_flip.png'), X2)
    cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_flip.png'), Y)

# Blurring images
for X1_path in os.listdir(os.path.join(destination_folder, 'image_2')):
    print(f'Blurring images {X1_path}')
    X1_path = X1_path.split('.')[0]
    X2_path = X1_path.split('.')[0]
    Y_path = X1_path.split('.')[0]

    X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'))
    X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'))
    Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))

    X1 = cv2.blur(X1, (blur_kernel, blur_kernel))
    X2 = cv2.blur(X2, (blur_kernel, blur_kernel))

    cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_blur.png'), X1)
    cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_blur.png'), X2)
    cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_blur.png'), Y)

# Scaling images
for X1_path in os.listdir(os.path.join(destination_folder, 'image_2')):
    print(f'Scaling images {X1_path}')
    X1_path = X1_path.split('.')[0]
    X2_path = X1_path.split('.')[0]
    Y_path = X1_path.split('.')[0]

    X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'))
    X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'))
    Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))

    # 384, 1248 for width and height of KITTI 2015 images
    X1 = cv2.resize(X1, (384 * scale_ratio, 1248 * scale_ratio))
    X2 = cv2.resize(X2, (384 * scale_ratio, 1248 * scale_ratio))
    Y = cv2.resize(Y, (384 * scale_ratio, 1248 * scale_ratio))

    cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_scale.png'), X1)
    cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_scale.png'), X2)
    cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_scale.png'), Y)

# Add Gaussian noises
for X1_path in os.listdir(os.path.join(destination_folder, 'image_2')):
    print(f'Add Gaussian noises {X1_path}')
    X1_path = X1_path.split('.')[0]
    X2_path = X1_path.split('.')[0]
    Y_path = X1_path.split('.')[0]

    X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'))
    X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'))
    Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))

    noise_mask = np.random.random(X1.shape[:2]) <= gaussian_noise_ratio
    noise = np.random.normal(gaussian_noise_mean, gaussian_noise_std, X1.shape[:2]) * 255
    for i in range(3):
        X1[..., i][noise_mask] = X1[..., i].astype('float64')[noise_mask] + noise[noise_mask]
    X1 = X1.astype('uint8')

    noise_mask = np.random.random(X2.shape[:2]) <= gaussian_noise_ratio
    noise = np.random.normal(gaussian_noise_mean, gaussian_noise_std, X2.shape[:2]) * 255
    for i in range(3):
        X2[..., i][noise_mask] = X2[..., i].astype('float64')[noise_mask] + noise[noise_mask]
    X2 = X2.astype('uint8')

    cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_noise.png'), X1)
    cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_noise.png'), X2)
