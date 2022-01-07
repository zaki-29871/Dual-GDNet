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


version = None
max_version = 2000  # KITTI 2015 v1497 recommended version
batch = 1
seed = 0
loss_threshold = 10
full_dataset = True
small_dataset = False
is_plot_image = False
untexture_rate = 0
dataset = ['flyingthings3D', 'KITTI_2015']
image = ['cleanpass', 'finalpass']  # for flyingthings3D
exception_count = 0

used_profile = profile.GDNet_sdc6f()
dataset = dataset[0]
if dataset == 'flyingthings3D':
    image = image[1]

# GTX 1660 Ti
if isinstance(used_profile, profile.GDNet_sdc6f):
    height, width = 192, 576  # 576 - 192 = 384
    max_disparity = 192

elif isinstance(used_profile, profile.GDNet_mdc6f):
    height, width = 192, 576  # 576 - 144 = 432
    max_disparity = 144

elif isinstance(used_profile, profile.GDNet_fdc6f):
    height, width = 96, 320  # 320 - 128 = 192
    max_disparity = 128

model = used_profile.load_model(max_disparity, version)[1]
version, loss_history = used_profile.load_history(version)

print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', used_profile)
print('Using dataset:', dataset)
print('Image size:', (height, width))
print('Max disparity:', max_disparity)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if dataset == 'flyingthings3D':
    train_dataset = FlyingThings3D(max_disparity, crop_size=(height, width), type='train', crop_seed=None, image=image,
                                   small=small_dataset)
    test_dataset = FlyingThings3D(max_disparity, crop_size=(height, width), type='test', crop_seed=None,
                                  small=small_dataset)

    if not full_dataset:
        train_dataset = random_subset(train_dataset, 1920, seed=seed)
        test_dataset = random_subset(test_dataset, 480, seed=seed)

elif dataset == 'KITTI_2015':
    train_dataset, test_dataset = random_split(
        KITTI_2015(max_disparity, crop_size=(height, width), type='train', crop_seed=None,
                   untexture_rate=untexture_rate), seed=seed)
else:
    raise Exception('Cannot find dataset: ' + dataset)

print('Number of training data:', len(train_dataset))
print('Number of testing data:', len(test_dataset))
os.system('nvidia-smi')

# 5235 MB
v = version
while v < max_version + 1:
    try:
        epoch_start_time = datetime.datetime.now()
        print('Exception count:', exception_count)
        if dataset == 'flyingthings3D':
            if isinstance(used_profile, profile.GDNet_fdc6f):
                train_loader = DataLoader(random_subset(train_dataset, 72), batch_size=batch, shuffle=False)
                test_loader = DataLoader(random_subset(test_dataset, 8), batch_size=batch, shuffle=False)
            else:
                train_loader = DataLoader(random_subset(train_dataset, 192), batch_size=batch, shuffle=False)
                test_loader = DataLoader(random_subset(test_dataset, 48), batch_size=batch, shuffle=False)

        elif dataset == 'KITTI_2015':
            train_loader = DataLoader(random_subset(train_dataset, 160), batch_size=batch, shuffle=False)
            test_loader = DataLoader(random_subset(test_dataset, 40), batch_size=batch, shuffle=False)
        else:
            raise Exception('Cannot find dataset: ' + dataset)

        train_loss = []
        test_loss = []
        error = []
        total_eval = []

        print('Start training, version = {}'.format(v))
        model.train()
        for batch_index, (X, Y) in enumerate(train_loader):
            if torch.all(Y == 0):
                print('Detect Y are all zero')
                continue
            utils.tic()
            if isinstance(used_profile, profile.GDNet_flip_training):
                optimizer.zero_grad()
                train_dict0 = used_profile.train(X, Y, dataset, flip=False)
                train_dict0['loss'].backward()
                optimizer.step()

                optimizer.zero_grad()
                train_dict1 = used_profile.train(X, Y, dataset, flip=True)
                train_dict1['loss'].backward()
                optimizer.step()

                wl = width / (2 * width - max_disparity)
                wr = (width - max_disparity) / (2 * width - max_disparity)

                loss = wl * train_dict0['loss'] + wr * train_dict1['loss']
                epe_loss = wl * train_dict0['epe_loss'] + wr * train_dict1['epe_loss']
                train_dict = train_dict0
            else:
                optimizer.zero_grad()
                train_dict = used_profile.train(X, Y, dataset)
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
                plotter.plot_image_disparity(X[0], Y[0, 0], dataset, train_dict,
                                             max_disparity=max_disparity)

            if torch.isnan(loss):
                print('detect loss nan in training')
                exit(0)

        train_loss = float(torch.tensor(train_loss).mean())
        print(f'Avg train loss = {utils.threshold_color(train_loss)}{train_loss:.3f}{Style.RESET_ALL}')

        print('Start testing, version = {}'.format(v))
        model.eval()
        for batch_index, (X, Y) in enumerate(test_loader):
            if torch.all(Y == 0):
                print('Detect Y are all zero')
                continue
            with torch.no_grad():
                utils.tic()
                if isinstance(used_profile, profile.GDNet_mdc6):
                    eval_dict = used_profile.eval(X, Y, dataset, merge_cost=False, lr_check=False, candidate=False,
                                                  regression=True,
                                                  penalize=False, slope=1, max_disparity_diff=1.5)
                else:
                    eval_dict = used_profile.eval(X, Y, dataset)

                time = utils.timespan_str(utils.toc(True))
                loss_str = f'epe loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
                error_rate_str = f'error rate = {eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
                print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, {error_rate_str}')

                test_loss.append(float(eval_dict["epe_loss"]))
                error.append(float(eval_dict["error_sum"]))
                total_eval.append(float(eval_dict["total_eval"]))

                if is_plot_image:
                    plotter = utils.CostPlotter()
                    plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict,
                                                 max_disparity=max_disparity)

                if torch.isnan(eval_dict["epe_loss"]):
                    print('detect loss nan in testing')
                    exit(0)

        test_loss = float(torch.tensor(test_loss).mean())
        test_error_rate = np.array(error).sum() / np.array(total_eval).sum()
        loss_str = f'epe loss = {utils.threshold_color(test_loss)}{test_loss:.3f}{Style.RESET_ALL}'
        error_rate_str = f'error rate = {test_error_rate:.2%}'
        print(f'Avg {loss_str}, {error_rate_str}')

        loss_history['train'].append(train_loss)
        loss_history['test'].append(test_loss)

        print('Start save model')
        used_profile.save_version(model, loss_history, v)
        epoch_end_time = datetime.datetime.now()
        print(f'[{utils.timespan_str(epoch_end_time - epoch_start_time)}] version = {v}')
        v += 1

    except Exception as err:
        # traceback.format_exc()  # Traceback string
        traceback.print_exc()
        exception_count += 1
        v -= 1
        if exception_count >= 50:
            exit(-1)
        # exit(-1)
