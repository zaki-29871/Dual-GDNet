from dataset import *
from torch.utils.data import DataLoader
import numpy as np
from profile import *
from colorama import Style
import profile

max_disparity = 192
# max_disparity = 144
version = None
seed = 0
lr_check = False
max_disparity_diff = 1.5
merge_cost = True
candidate = False
plot_and_save_image = True
use_resize = True
use_split_prduce_disparity = False
use_margin_prduce_disparity = False

resize_height, resize_width = 352, 960  # KITTI, 2015 GTX 1660 Ti
split_height, split_width = 352, 960  # KITTI, 2015 GTX 1660 Ti
# split_height, split_width = 192, 1216  # KITTI 2015 GTX 1660 Ti
margin_height, margin_width = 384, 1248  # KITTI 2015 GTX 1660 Ti

assert use_split_prduce_disparity ^ use_margin_prduce_disparity ^ use_resize, 'Just can use one setting'

used_profile = profile.GDNet_mdc6f()
dataset = 'KITTI_2015_benchmark'

model = used_profile.load_model(max_disparity, version)[1]
version, loss_history = used_profile.load_history(version)

print('Using model:', used_profile)
print('Using dataset:', dataset)
# print('Image size:', (KITTI_2015_benchmark.HEIGHT, KITTI_2015_benchmark.WIDTH))
print('Max disparity:', max_disparity)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))
print('Using split produce disparity mode:', use_split_prduce_disparity)
print('Using large margin produce disparity mode:', use_margin_prduce_disparity)

losses = []
error = []
confidence_error = []
total_eval = []

test_dataset = KITTI_2015_benchmark(use_resize=use_resize, resize_height=resize_height, resize_width=resize_width)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Number of testing data:', len(test_dataset))

model.eval()
for batch_index, (X, Y, origin_height, origin_width) in enumerate(test_loader):
    with torch.no_grad():
        utils.tic()

        if use_split_prduce_disparity:
            eval_dict = utils.split_prduce_disparity(used_profile, X, Y, dataset, max_disparity, split_height,
                                                     split_width, merge_cost=merge_cost,
                                                     lr_check=False,
                                                     candidate=candidate,
                                                     regression=True,
                                                     penalize=False, slope=1, max_disparity_diff=1.5)
        elif use_margin_prduce_disparity:
            eval_dict = used_profile.eval_cpu(X, Y, dataset, margin_height, margin_width, margin_full=0xff,
                                              merge_cost=merge_cost)
        else:
            eval_dict = used_profile.eval(X, Y, dataset, merge_cost=merge_cost, lr_check=False, candidate=candidate,
                                          regression=True,
                                          penalize=False, slope=1, max_disparity_diff=1.5)

        time = utils.timespan_str(utils.toc(True))
        print(f'[{batch_index + 1}/{len(test_loader)} {time}]')

        losses.append(float(eval_dict["epe_loss"]))
        error.append(float(eval_dict["error_sum"]))
        total_eval.append(float(eval_dict["total_eval"]))

        if merge_cost:
            confidence_error.append(float(eval_dict["CE_avg"]))

        if torch.isnan(eval_dict["epe_loss"]):
            print('detect loss nan in testing')
            exit(1)

        if plot_and_save_image:
            plotter = utils.CostPlotter()

            origin_width = int(origin_width)
            origin_height = int(origin_height)

            # dimension X: batch, channel*2, height, width = 1, 6, 352, 1216
            plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict,
                                         max_disparity=max_disparity,
                                         save_result_file=(f'{used_profile}_benchmark/{dataset}', batch_index, False,
                                                           None),
                                         is_benchmark=True)
        # exit(0)
        # os.system('nvidia-smi')

print(f'avg loss = {np.array(losses).mean():.3f}')
print(f'std loss = {np.array(losses).std():.3f}')
print(f'avg error rates = {np.array(error).sum() / np.array(total_eval).sum():.2%}')
if merge_cost:
    print(f'avg confidence error = {np.array(confidence_error).mean():.3f}')
print('Number of test case:', len(losses))
