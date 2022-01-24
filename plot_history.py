import matplotlib.pyplot as plt
import profile
import numpy as np
import utils
from colorama import Fore, Style

class EPE_Loss:
    def __init__(self):
        self.SGM = 7.405
        self.MC_CNN = 3.79
        self.GCNet = 2.51
        self.PSMNet = 1.09
        self.GANet = 0.84
        self.GANetSmall = 6.478
        self.CSPN = 0.78

def get_color(value):
    if value < 0:
        return Fore.GREEN
    else:
        return Fore.RED


version = None
trend_kernel = 10  # version (plot) + trend kernel = real model version, trend_kernel = [1, n]
trend_regression_size = 30  # to see the loss is decent or not, trend_regression_size = [1, n]
trend_method = ['corr', 'regression'][1]
epe = EPE_Loss()
used_profile = profile.GDNet_sdc6f()
start_version = 800  # start_version = [1, n]

version, loss_history = used_profile.load_history(version)
print('Number of epochs:', len(loss_history['test']))
print('Trend kernel size:', trend_kernel)
print('Trend regression size:', trend_regression_size)

if len(loss_history['test']) == 1:
    marker = 'o'
else:
    marker = 'o'

plt.figure()
plt.title('EPE Loss')
plt.xlabel('Epoch')
plt.ylabel('EPE Loss')

assert start_version >= 1 and isinstance(start_version, int)
train_loss_history = loss_history['train'][start_version - 1:]
test_loss_history = loss_history['test'][start_version - 1:]
print('Size of loss history:', len(train_loss_history))

p_train = plt.plot(train_loss_history[(trend_kernel - 1):], label='Train', marker=marker)
p_test = plt.plot(test_loss_history[(trend_kernel - 1):], label='Test', marker=marker)

if trend_kernel > 1:
    test_loss_trend = np.array(test_loss_history)
    test_loss_trend = np.convolve(test_loss_trend, np.ones(trend_kernel, )) / trend_kernel
    test_loss_trend = test_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
    # plt.plot(test_loss_trend, label='Test Trend', marker=marker, color=p_train[0].get_color(), linestyle='--')
    plt.plot(test_loss_trend, label='Test Trend', marker=marker)

    train_loss_trend = np.array(train_loss_history)
    train_loss_trend = np.convolve(train_loss_trend, np.ones(trend_kernel, )) / trend_kernel
    train_loss_trend = train_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
    # plt.plot(train_loss_trend, label='Train Trend', marker=marker, color=p_test[0].get_color(), linestyle='--')
    plt.plot(train_loss_trend, label='Train Trend', marker=marker)

    print('Trend method:', trend_method)

    train_loss_trend_regression = utils.trend_regression(train_loss_trend[-trend_regression_size:], method=trend_method)
    test_loss_trend_regression = utils.trend_regression(test_loss_trend[-trend_regression_size:], method=trend_method)
    test_train_diff = test_loss_trend[-1] - train_loss_trend[-1]
    train_loss_trend_color = None
    test_loss_trend_color = None
    test_train_diff_color = None

    print(f'Train loss trend: {get_color(train_loss_trend_regression)}{train_loss_trend_regression:.2e}{Style.RESET_ALL}')
    print(f'Test loss trend: {get_color(test_loss_trend_regression)}{test_loss_trend_regression:.2e}{Style.RESET_ALL}')
    print(f'Last test loss - train loss (large is overfitting): {get_color(test_train_diff)}{test_train_diff:.2e}{Style.RESET_ALL}')

# plt.axhline(epe.SGM, color='b', linestyle='--', label='SGM (320 images)')
# plt.axhline(epe.MC_CNN, color='g', linestyle='--', label='MC_CNN')
# plt.axhline(epe.GCNet, color='r', linestyle='--', label='GCNet')
plt.axhline(epe.PSMNet, color='c', linestyle='--', label='PSMNet')
plt.axhline(epe.GANet, color='m', linestyle='--', label='GANet-15')
# plt.axhline(epe.GANetSmall, color='y', linestyle='--', label='GANet-small (320 images)')
plt.axhline(epe.CSPN, color='k', linestyle='--', label='3DCSPN_ds_ss + CSPF')

plt.legend()
plt.show()
