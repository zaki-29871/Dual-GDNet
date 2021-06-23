from dataset import FlyingThings3D, random_subset, random_split, KITTI_2015, AerialImagery
from torch.utils.data import DataLoader
import numpy as np
from profile import *
from colorama import Style
import profile

# max_disparity = 192
max_disparity = 144
# version = 592
version = None
seed = 0
lr_check = False
max_disparity_diff = 1.5
merge_cost = False
candidate = False
dataset = ['flyingthings3D', 'KITTI_2015', 'KITTI_2015_benchmark', 'AerialImagery']
image = ['cleanpass', 'finalpass']  # for flyingthings3D

used_profile = profile.GDNet_mdc6()
dataset = dataset[0]
image = image[1]

if dataset == 'flyingthings3D':
    # height, width = 512, 960
    height, width = 384, 960

elif dataset == 'KITTI_2015':
    height, width = 352, 1216
    # height, width = 336, 1200  # GDNet_dc6f

elif dataset == 'KITTI_2015_benchmark':
    height, width = 352, 1216
    # height, width = 336, 1200  # GDNet_dc6f

elif dataset == 'AerialImagery':
    height, width = AerialImagery.image_size

else:
    height = None
    width = None
    raise Exception('Cannot find dataset: ' + dataset)

model = used_profile.load_model(max_disparity, version)[1]
version, loss_history = used_profile.load_history(version)

print('Using model:', used_profile)
print('Using dataset:', dataset)
print('Image size:', (height, width))
print('Max disparity:', max_disparity)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

losses = []
error = []
confidence_error = []
total_eval = []

if dataset == 'flyingthings3D':
    test_dataset = FlyingThings3D((height, width), max_disparity, type='test', crop_seed=0, image=image)
    test_dataset = random_subset(test_dataset, 100, seed=seed)

elif dataset == 'KITTI_2015':
    train_dataset, test_dataset = random_split(KITTI_2015((height, width), type='train', crop_seed=0, untexture_rate=0), seed=seed)

elif dataset == 'KITTI_2015_benchmark':
    test_dataset = KITTI_2015((height, width), type='test', crop_seed=0)

elif dataset == 'AerialImagery':
    test_dataset = AerialImagery()

else:
    raise Exception('Cannot find dataset: ' + dataset)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('Number of testing data:', len(test_dataset))
model.eval()
for batch_index, (X, Y) in enumerate(test_loader):
    with torch.no_grad():
        utils.tic()

        if isinstance(used_profile, profile.GDNet_mdc6):
            eval_dict = used_profile.eval(X, Y, dataset, merge_cost=merge_cost, lr_check=False, candidate=candidate, regression=True,
                                     penalize=False, slope=1, max_disparity_diff=1.5)
        else:
            eval_dict = used_profile.eval(X, Y, dataset)


        time = utils.timespan_str(utils.toc(True))
        loss_str = f'loss = {utils.threshold_color(eval_dict["epe_loss"])}{eval_dict["epe_loss"]:.3f}{Style.RESET_ALL}'
        error_rate_str = f'{eval_dict["error_sum"] / eval_dict["total_eval"]:.2%}'
        print(f'[{batch_index + 1}/{len(test_loader)} {time}] {loss_str}, error rate = {error_rate_str}')

        losses.append(float(eval_dict["epe_loss"]))
        error.append(float(eval_dict["error_sum"]))
        total_eval.append(float(eval_dict["total_eval"]))

        if merge_cost:
            confidence_error.append(float(eval_dict["CE_avg"]))

        if torch.isnan(eval_dict["epe_loss"]):
            print('detect loss nan in testing')
            exit(1)

        plotter = utils.CostPlotter()

        plotter.plot_image_disparity(X[0], Y[0, 0], dataset, eval_dict,
                                     max_disparity=max_disparity,
                                     save_result_file=(f'{used_profile}/{dataset}', batch_index, False,
                                                       error_rate_str))
        # exit(0)
        # os.system('nvidia-smi')

print(f'avg loss = {np.array(losses).mean():.3f}')
print(f'std loss = {np.array(losses).std():.3f}')
print(f'avg error rates = {np.array(error).sum() / np.array(total_eval).sum():.2%}')
if merge_cost:
    print(f'avg confidence error = {np.array(confidence_error).mean():.3f}')
print('Number of test case:', len(losses))
