import numpy as np
import cv2
import os
import shutil

# Setting
original_data_folder = r'F:\Dataset\KITTI 2015\training'
destination_folder = r'F:\Dataset\KITTI 2015 Data Augmentation'
copy_size = 200
# blur_kernels = [3]
# scale_ratios = [1.5]
# gaussian_noises = [(0.05, 0, 0.05)]  # ratio [0, 1], mean, std
blur_kernels = [3, 5]
scale_ratios = [0.8, 1.2, 1.5, 2]
gaussian_noises = [(0.3, 0, 0.075), (0.5, 0, 0.05)]  # ratio [0, 1], mean, std
count = 1

if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder, exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'image_2'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'image_3'), exist_ok=True)
os.makedirs(os.path.join(destination_folder, 'disp_occ_0'), exist_ok=True)

augmentation_multiplier = 2 * (len(blur_kernels) + 1) * (len(scale_ratios) + 1) * (len(gaussian_noises) + 1)
total_data_size = copy_size * augmentation_multiplier
print('Original data size:', copy_size)
print('Augmentation multiplier:', augmentation_multiplier)
print('Augmented total data size:', copy_size * augmentation_multiplier)

# Copy original images
for i in range(0, copy_size):
    print(f'[{count}/{total_data_size} {count/total_data_size:.0%}] Copy index {i}')
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

    count += 1

# Vertical flipping images
for X1_path_original in os.listdir(os.path.join(destination_folder, 'image_2')):
    print(f'[{count}/{total_data_size} {count/total_data_size:.0%}] Vertical flipping {X1_path_original}')
    X1_path = X1_path_original.split('.')[0]
    X2_path = X1_path_original.split('.')[0]
    Y_path = X1_path_original.split('.')[0]

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

    count += 1

# Blurring images
for X1_path_original in os.listdir(os.path.join(destination_folder, 'image_2')):
    for blur_kernel in blur_kernels:
        print(f'[{count}/{total_data_size} {count/total_data_size:.0%}] Blurring images {X1_path_original}, blur kernel = {blur_kernel}')
        X1_path = X1_path_original.split('.')[0]
        X2_path = X1_path_original.split('.')[0]
        Y_path = X1_path_original.split('.')[0]

        X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'))
        X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'))
        Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))

        X1 = cv2.blur(X1, (blur_kernel, blur_kernel))
        X2 = cv2.blur(X2, (blur_kernel, blur_kernel))

        cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_blur_{blur_kernel}.png'), X1)
        cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_blur_{blur_kernel}.png'), X2)
        cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_blur_{blur_kernel}.png'), Y)

        count += 1
# Scaling images
for X1_path_original in os.listdir(os.path.join(destination_folder, 'image_2')):
    for scale_ratio in scale_ratios:
        print(f'[{count}/{total_data_size} {count/total_data_size:.0%}] Scaling images {X1_path_original}, scale_ratio = {scale_ratio}')
        X1_path = X1_path_original.split('.')[0]
        X2_path = X1_path_original.split('.')[0]
        Y_path = X1_path_original.split('.')[0]

        X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png'))
        X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png'))
        Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))
        resize_width, resize_hight = int(1248 * scale_ratio), int(384 * scale_ratio)

        # 384, 1248 for width and height of KITTI 2015 images
        X1 = cv2.resize(X1, (resize_width, resize_hight))
        X2 = cv2.resize(X2, (resize_width, resize_hight))
        Y = cv2.resize(Y, (resize_width, resize_hight))

        scale_ratio_str = str(scale_ratio).replace('.', '-')
        cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_scale_{scale_ratio_str}.png'), X1)
        cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_scale_{scale_ratio_str}.png'), X2)
        cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_scale_{scale_ratio_str}.png'), Y)

        count += 1
# Add Gaussian noises
for X1_path_original in os.listdir(os.path.join(destination_folder, 'image_2')):
    for ratio, mean, std in gaussian_noises:
        print(f'[{count}/{total_data_size} {count/total_data_size:.0%}] Add Gaussian noises {X1_path_original}, ratio = {ratio}, mean = {mean}, std = {std}')
        X1_path = X1_path_original.split('.')[0]
        X2_path = X1_path_original.split('.')[0]
        Y_path = X1_path_original.split('.')[0]

        X1 = cv2.imread(os.path.join(destination_folder, 'image_2', f'{X1_path}.png')).astype('float64')
        X2 = cv2.imread(os.path.join(destination_folder, 'image_3', f'{X2_path}.png')).astype('float64')
        Y = cv2.imread(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}.png'))

        noise_mask = np.random.random(X1.shape[:2]) <= ratio
        noise = np.random.normal(mean, std, X1.shape[:2]) * 255
        for i in range(3):
            X1[..., i][noise_mask] = X1[..., i][noise_mask] + noise[noise_mask]
        X1[X1 > 255] = 255
        X1[X1 < 0] = 0
        X1 = X1.astype('uint8')

        noise_mask = np.random.random(X2.shape[:2]) <= ratio
        noise = np.random.normal(mean, std, X2.shape[:2]) * 255
        for i in range(3):
            X2[..., i][noise_mask] = X2[..., i][noise_mask] + noise[noise_mask]
        X2[X2 > 255] = 255
        X2[X2 < 0] = 0
        X2 = X2.astype('uint8')

        gaussian_noise_str = str(ratio).replace('.', '-') + '_' + str(mean).replace('.', '-') + '_' + str(std).replace(
            '.', '-')
        cv2.imwrite(os.path.join(destination_folder, 'image_2', f'{X1_path}_noise_{gaussian_noise_str}.png'), X1)
        cv2.imwrite(os.path.join(destination_folder, 'image_3', f'{X2_path}_noise_{gaussian_noise_str}.png'), X2)
        cv2.imwrite(os.path.join(destination_folder, 'disp_occ_0', f'{Y_path}_noise_{gaussian_noise_str}.png'), Y)

        count += 1