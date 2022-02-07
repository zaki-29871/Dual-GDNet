from torch.utils.data import DataLoader
from dataset.dataset import *

def make_data_loader(max_disparity, height, width, **kwargs):
    train_dataset = FlyingThings3D(max_disparity, type='train', use_crop_size=True, crop_size=(height, width),
                                   crop_seed=None, image='finalpass', **kwargs)
    test_dataset = FlyingThings3D(max_disparity, type='test', use_crop_size=True, crop_size=(height, width),
                                  crop_seed=None, image='finalpass', **kwargs)
    return train_dataset, test_dataset