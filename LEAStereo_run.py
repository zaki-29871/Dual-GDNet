import torch.nn.functional as F
from LEAStereo.LEAStereo import LEAStereo
from LEAStereo.make_data_loader import make_data_loader
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset.dataset import *
from colorama import Style
import os
import utils
import datetime

def main():
    kwargs = {'num_workers': 8, 'pin_memory': True, 'drop_last': True}

    version = None
    max_version = 2000  # KITTI 2015 v1497 recommended version
    batch = 4
    seed = 0
    loss_threshold = 10
    is_plot_image = False
    untexture_rate = 0
    dataset = ['flyingthings3D', 'KITTI_2015', 'KITTI_2015_Augmentation', 'KITTI_2012_Augmentation']
    exception_count = 0
    dataset = dataset[0]

    # GTX 1660 Ti
    # height, width, max_disparity = 288, 576, 192
    height, width, max_disparity = 144, 288, 120

    model = LEAStereo(maxdisp=max_disparity, maxdisp_downsampleing=3).cuda()
    # model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if dataset == 'flyingthings3D':
        train_dataset = FlyingThings3D(max_disparity, type='train', use_crop_size=True, crop_size=(height, width),
                                       crop_seed=None, image='finalpass')
        test_dataset = FlyingThings3D(max_disparity, type='test', use_crop_size=True, crop_size=(height, width),
                                      crop_seed=None, image='finalpass')

    elif dataset == 'KITTI_2015':
        train_dataset, test_dataset = random_split(
            KITTI_2015(use_crop_size=True, crop_size=(height, width), type='train', crop_seed=None,
                       untexture_rate=untexture_rate), seed=seed)

    elif dataset == 'KITTI_2015_Augmentation':
        train_dataset = KITTI_2015_Augmentation(use_crop_size=True, crop_size=(height, width), type='train',
                                                crop_seed=None,
                                                shuffle_seed=0)
        test_dataset = KITTI_2015_Augmentation(use_crop_size=True, crop_size=(height, width), type='test',
                                               crop_seed=None,
                                               shuffle_seed=0)

    elif dataset == 'KITTI_2012_Augmentation':
        train_dataset = KITTI_2012_Augmentation(use_crop_size=True, crop_size=(height, width), type='train',
                                                crop_seed=None,
                                                shuffle_seed=0)
        test_dataset = KITTI_2012_Augmentation(use_crop_size=True, crop_size=(height, width), type='test',
                                               crop_seed=None,
                                               shuffle_seed=0)

    else:
        raise Exception('Cannot find dataset: ' + dataset)

    print('Number of training data:', len(train_dataset))
    print('Number of testing data:', len(test_dataset))
    os.system('nvidia-smi')

    # 5235 MB
    v = 1
    while v < max_version + 1:
        print('version:', version)
        epoch_start_time = datetime.datetime.now()
        print('Exception count:', exception_count)
        if dataset == 'flyingthings3D':
            train_loader = DataLoader(random_subset(train_dataset, 576), batch_size=batch, shuffle=False, **kwargs)
            test_loader = DataLoader(random_subset(test_dataset, 144), batch_size=batch, shuffle=False, **kwargs)

        elif dataset == 'KITTI_2015':
            train_loader = DataLoader(random_subset(train_dataset, 160), batch_size=batch, shuffle=False, **kwargs)
            test_loader = DataLoader(random_subset(test_dataset, 40), batch_size=batch, shuffle=False, **kwargs)

        elif dataset in ['KITTI_2015_Augmentation', 'KITTI_2012_Augmentation']:
            train_loader = DataLoader(random_subset(train_dataset, 576), batch_size=batch, shuffle=False, **kwargs)
            test_loader = DataLoader(random_subset(test_dataset, 144), batch_size=batch, shuffle=False, **kwargs)

        else:
            raise Exception('Cannot find dataset: ' + dataset)

        train_loss = []
        test_loss = []
        error = []
        total_eval = []

        print('Start training, version = {}'.format(v))
        model.train()
        for batch_index, (X, Y) in enumerate(train_loader):
            X, Y = X.cuda(), Y.cuda()
            if torch.all(Y == 0):
                print('Detect Y are all zero')
                continue
            utils.tic()

            optimizer.zero_grad()
            mask = utils.y_mask(Y, max_disparity, dataset)
            disp = model(X[:, 0:3, :, :], X[:, 3:6, :, :])

            loss = F.smooth_l1_loss(disp.unsqueeze(1)[mask], Y[mask], reduction='mean')
            loss.backward()
            optimizer.step()

            time = utils.timespan_str(utils.toc(True))
            loss_str = f'loss = {utils.threshold_color(loss)}{loss:.3f}{Style.RESET_ALL}'
            print(f'[{batch_index + 1}/{len(train_loader)} {time}] {loss_str}')

if __name__ == '__main__':
    main()