from torch.utils.data import DataLoader
import torch.optim as optim
from dataset.dataset import *
from colorama import Style
import profile
import numpy as np
import os
import utils
import traceback
import datetime
import torch.nn.functional as F


def main():
    version = None
    cv_model = None
    max_version = 2000
    batch = 1
    seed = 0
    is_plot_image = False
    is_debug = False
    dataset_name = ['flyingthings3D', 'KITTI_2015', 'KITTI_2015_Augmentation', 'KITTI_2012_Augmentation'][0]
    exception_count = 0
    used_cv_profile = profile.GDNet_sdc6f()
    used_disp_profile = profile.MergeNet_d()
    dataloader_kwargs = {'num_workers': 8, 'pin_memory': True, 'drop_last': True}

    # GTX 1660 Ti
    if isinstance(used_cv_profile, profile.GDNet_sdc6f):
        height, width = 192, 576  # 576 - 192 = 384
        max_disparity = 192

        # height, width = 128, 384  # 384 - 128 = 256
        # max_disparity = 128

    elif isinstance(used_cv_profile, (profile.GDNet_sd9c6, profile.GDNet_sd9c6f)):
        height, width = 128, 384  # 384 - 128 = 256
        max_disparity = 128

    elif isinstance(used_cv_profile, profile.GDNet_mdc6f):
        height, width = 192, 576  # 576 - 144 = 432
        max_disparity = 144

    elif isinstance(used_cv_profile, profile.GDNet_fdc6f):
        height, width = 96, 320  # 320 - 128 = 192
        max_disparity = 128

    elif isinstance(used_cv_profile, profile.LEAStereo_fdcf):
        height, width = 240, 576  # 576 - 150 = 426
        max_disparity = 150

    elif isinstance(used_cv_profile, profile.GDNet_sd9d6):
        height, width = 128, 448  # 448 - 160 = 288
        max_disparity = 160


    cv_model = used_cv_profile.load_model(max_disparity, cv_model)[1]
    disp_model = used_disp_profile.load_model(max_disparity, version)[1]
    version, loss_history = used_disp_profile.load_history(version)
    torch.backends.cudnn.benchmark = True

    print(f'CUDA abailable cores: {torch.cuda.device_count()}')
    print(f'Batch: {batch}')
    print('Using model:', used_disp_profile)
    print('Using dataset:', dataset_name)
    print('Image size:', (height, width))
    print('Max disparity:', max_disparity)
    print('Number of parameters: {:,}'.format(sum(p.numel() for p in disp_model.parameters())))

    optimizer = optim.Adam(disp_model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if dataset_name == 'flyingthings3D':
        train_dataset = FlyingThings3D(max_disparity, type='train', use_crop_size=True, crop_size=(height, width),
                                       crop_seed=None, image='finalpass')
        test_dataset = FlyingThings3D(max_disparity, type='test', use_crop_size=True, crop_size=(height, width),
                                      crop_seed=None, image='finalpass')

    elif dataset_name == 'KITTI_2015':
        train_dataset, test_dataset = random_split(
            KITTI_2015(use_crop_size=True, crop_size=(height, width), type='train', crop_seed=None), seed=seed)

    elif dataset_name == 'KITTI_2015_Augmentation':
        train_dataset = KITTI_2015_Augmentation(use_crop_size=True, crop_size=(height, width), type='train',
                                                crop_seed=None,
                                                shuffle_seed=0)
        test_dataset = KITTI_2015_Augmentation(use_crop_size=True, crop_size=(height, width), type='test',
                                               crop_seed=None,
                                               shuffle_seed=0)

    elif dataset_name == 'KITTI_2012_Augmentation':
        train_dataset = KITTI_2012_Augmentation(use_crop_size=True, crop_size=(height, width), type='train',
                                                crop_seed=None,
                                                shuffle_seed=0)
        test_dataset = KITTI_2012_Augmentation(use_crop_size=True, crop_size=(height, width), type='test',
                                               crop_seed=None,
                                               shuffle_seed=0)

    else:
        raise Exception('Cannot find dataset: ' + dataset_name)

    print('Number of training data:', len(train_dataset))
    print('Number of testing data:', len(test_dataset))

    v = version
    while v < max_version + 1:
        try:
            epoch_start_time = datetime.datetime.now()
            print('Exception count:', exception_count)
            if dataset_name == 'flyingthings3D':
                # 960, 240
                train_loader = DataLoader(random_subset(train_dataset, 192), batch_size=batch, shuffle=False,
                                          **dataloader_kwargs)
                test_loader = DataLoader(random_subset(test_dataset, 48), batch_size=batch, shuffle=False,
                                         **dataloader_kwargs)

            elif dataset_name == 'KITTI_2015':
                train_loader = DataLoader(random_subset(train_dataset, 160), batch_size=batch, shuffle=False,
                                          **dataloader_kwargs)
                test_loader = DataLoader(random_subset(test_dataset, 40), batch_size=batch, shuffle=False,
                                         **dataloader_kwargs)

            elif dataset_name in ['KITTI_2015_Augmentation', 'KITTI_2012_Augmentation']:
                train_loader = DataLoader(random_subset(train_dataset, 960), batch_size=batch, shuffle=False,
                                          **dataloader_kwargs)
                test_loader = DataLoader(random_subset(test_dataset, 240), batch_size=batch, shuffle=False,
                                         **dataloader_kwargs)

            else:
                raise Exception('Cannot find dataset: ' + dataset_name)

            train_loss = []
            test_loss = []
            error = []
            total_eval = []

            print('Start training')
            cv_model.eval()
            disp_model.train()
            for batch_index, (X, Y, pass_info) in enumerate(train_loader):
                if torch.all(Y == 0):
                    print('Detect Y are all zero')
                    continue
                X, Y = X.cuda(), Y.cuda()

                utils.tic()
                with torch.no_grad():
                    cv_model.flip = False
                    cost_left = cv_model(X[:, 0:3, :, :], X[:, 3:6, :, :]).unsqueeze(1)
                    cv_model.flip = True
                    cost_right = cv_model(X[:, 0:3, :, :], X[:, 3:6, :, :]).unsqueeze(1)

                optimizer.zero_grad()
                train_dict = used_disp_profile.train(cost_left, cost_right, Y, dataset_name)
                train_dict['loss'].backward()
                loss = train_dict['loss']
                epe_loss = train_dict['epe_loss']
                optimizer.step()

                train_loss.append(float(epe_loss))

                time = utils.timespan_str(utils.toc(True))
                loss_str = f'loss = {utils.threshold_color(loss)}{loss:.3f}{Style.RESET_ALL}'
                epe_loss_str = f'epe_loss = {utils.threshold_color(epe_loss)}{epe_loss:.3f}{Style.RESET_ALL}'
                print(f'[{batch_index + 1}/{len(train_loader)} {time}] {loss_str}, {epe_loss_str}')

                if is_plot_image:
                    plotter = utils.CostPlotter()
                    plotter.plot_image_disparity(X[0], Y[0, 0], dataset_name, train_dict,
                                                 max_disparity=max_disparity, use_resize=False,
                                                 use_padding_crop_size=False, pass_info=pass_info)

                if torch.isnan(loss):
                    raise Exception('detect loss nan in training')

            train_loss = float(torch.tensor(train_loss).mean())
            print(f'Avg train loss = {utils.threshold_color(train_loss)}{train_loss:.3f}{Style.RESET_ALL}')

            print('Start testing, version = {}'.format(v))
            disp_model.eval()
            for batch_index, (X, Y, pass_info) in enumerate(test_loader):
                if torch.all(Y == 0):
                    print('Detect Y are all zero')
                    continue
                X, Y = X.cuda(), Y.cuda()

                utils.tic()
                with torch.no_grad():
                    cv_model.flip = False
                    cost_left = cv_model(X[:, 0:3, :, :], X[:, 3:6, :, :]).unsqueeze(1)
                    cv_model.flip = True
                    cost_right = cv_model(X[:, 0:3, :, :], X[:, 3:6, :, :]).unsqueeze(1)
                    eval_dict = used_disp_profile.eval(cost_left, cost_right, Y, pass_info, dataset_name)
                    time = utils.timespan_str(utils.toc(True))
                    loss_str = f'epe loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
                    error_rate_str = f'error rate = {eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
                    print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, {error_rate_str}')

                    test_loss.append(float(eval_dict["epe_loss"]))
                    error.append(float(eval_dict["error_sum"]))
                    total_eval.append(float(eval_dict["total_eval"]))

                    if is_plot_image:
                        plotter = utils.CostPlotter()
                        plotter.plot_image_disparity(X[0], Y[0, 0], dataset_name, eval_dict,
                                                     max_disparity=max_disparity, use_resize=False,
                                                     use_padding_crop_size=False, pass_info=pass_info)

                    if torch.isnan(eval_dict["epe_loss"]):
                        raise Exception('detect loss nan in testing')

            test_loss = float(torch.tensor(test_loss).mean())
            test_error_rate = np.array(error).sum() / np.array(total_eval).sum()
            loss_str = f'epe loss = {utils.threshold_color(test_loss)}{test_loss:.3f}{Style.RESET_ALL}'
            error_rate_str = f'error rate = {test_error_rate:.2%}'
            print(f'Avg {loss_str}, {error_rate_str}')

            loss_history['train'].append(train_loss)
            loss_history['test'].append(test_loss)

            print('Start save model')
            used_disp_profile.save_version(disp_model, loss_history, v)
            epoch_end_time = datetime.datetime.now()
            print(f'[{utils.timespan_str(epoch_end_time - epoch_start_time)}] version = {v}')
            v += 1

        except Exception as err:
            # traceback.format_exc()  # Traceback string
            traceback.print_exc()
            exception_count += 1
            # if exception_count >= 50:
            #     exit(-1)
            if is_debug:
                exit(-1)


if __name__ == '__main__':
    main()
