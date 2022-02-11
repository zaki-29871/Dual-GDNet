import torch.nn as nn
import torch
import numpy as np


class MergeNet_d(nn.Module):
    def __init__(self, max_disparity):
        super(MergeNet_d, self).__init__()

        self.conv_11 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_12 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_13 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_14 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        # padding = 1/2 * kernel - 1
        self.conv_15 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_16 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_17 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_18 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.ReLU(inplace=True))

        self.conv_21 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_22 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_23 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_24 = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_25 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_26 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_27 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_28 = nn.Sequential(nn.ConvTranspose1d(1, 1, kernel_size=200, stride=2, padding=99),
                                     nn.BatchNorm1d(1),
                                     nn.LeakyReLU(inplace=True))

        self.conv_final = nn.Sequential(nn.Conv1d(1, 1, kernel_size=201, stride=2, padding=100, bias=False),
                                        nn.Conv1d(1, 1, kernel_size=201, stride=1, padding=100, bias=False),
                                        nn.Conv1d(1, 1, kernel_size=201, stride=1, padding=100, bias=False),
                                        nn.Conv1d(1, 1, kernel_size=201, stride=1, padding=100, bias=False),
                                        nn.Conv1d(1, 1, kernel_size=201, stride=1, padding=100, bias=False))

        self.disparity_regression = DisparityRegression(max_disparity)

    def forward(self, cost1, cost2):
        # Cost merging
        batch, disparityx2, height, width = cost1.size(0), cost1.size(2) * 2, cost1.size(3), cost1.size(4)
        cost = torch.cat((cost1, cost2), 1).reshape(batch, disparityx2, height, width)
        cost = cost.permute(0, 2, 3, 1)  # batch, height, width, disparity
        cost = cost.reshape(-1, 1, disparityx2)

        # Disparity computing
        rem0 = self.conv_11(cost)
        rem1 = self.conv_12(rem0)
        rem2 = self.conv_13(rem1)
        rem3 = self.conv_14(rem2)
        rem4 = self.conv_15(rem3)
        rem5 = self.conv_16(rem4)
        rem6 = self.conv_17(rem5)
        rem7 = self.conv_18(rem6)
        cost = self.conv_21(rem7)
        cost = self.conv_22(cost + rem0)
        cost = self.conv_23(cost + rem1)
        cost = self.conv_24(cost + rem2)
        cost = self.conv_25(cost + rem3)
        cost = self.conv_26(cost + rem4)
        cost = self.conv_27(cost + rem5)
        cost = self.conv_28(cost + rem6)
        disp = self.conv_final(cost + rem7)
        disp = self.disparity_regression(disp)
        disp = disp.reshape(batch, height, width)

        return disp


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        with torch.cuda.device_of(x):
            disp = np.arange(self.maxdisp).reshape([1, 1, self.maxdisp])
            disp = torch.tensor(disp, requires_grad=False).to(x.device).float()
            disp = disp.repeat(x.size(0), x.size(1), 1)
            disp = torch.sum(x * disp, dim=2)
        return disp
