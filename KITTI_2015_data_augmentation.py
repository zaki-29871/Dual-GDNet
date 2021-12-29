import numpy
import cv2
import os

# Setting
original_data_folder = r'F:\Dataset\KITTI 2015\training'
destination_folder = r'F:\Dataset\KITTI 2015 Data Augmentation'

os.makedirs(destination_folder, exist_ok=True)

for i in range(0, 201):
    X1 = cv2.imread(os.path.join(original_data_folder, 'image_2/{:06d}_10.png'.format(i)))
    X2 = cv2.imread(os.path.join(original_data_folder, 'image_3/{:06d}_10.png'.format(i)))
    Y = cv2.imread(os.path.join(original_data_folder, 'disp_occ_0/{:06d}_10.png'.format(i)))  # (376, 1241, 3) uint8
