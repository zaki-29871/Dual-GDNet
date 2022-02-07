import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from LEAStereo.models.build_model_2d import CostInterpolation
from LEAStereo.models.decoding_formulas import network_layer_to_space
from LEAStereo.new_model_2d import newFeature
from LEAStereo.skip_model_3d import newMatching


class LEAStereo(nn.Module):
    def __init__(self, maxdisp, maxdisp_downsampleing,
                 net_arch_fea='LEAStereo/run/architecture/feature_network_path.npy',
                 cell_arch_fea='LEAStereo/run/architecture/feature_genotype.npy',
                 net_arch_mat='LEAStereo/run/architecture/matching_network_path.npy',
                 cell_arch_mat='LEAStereo/run/architecture/matching_genotype.npy'):
        super(LEAStereo, self).__init__()

        network_path_fea, cell_arch_fea = np.load(net_arch_fea), np.load(cell_arch_fea)
        network_path_mat, cell_arch_mat = np.load(net_arch_mat), np.load(cell_arch_mat)
        print('Feature network path:{}\nMatching network path:{} \n'.format(network_path_fea, network_path_mat))

        network_arch_fea = network_layer_to_space(network_path_fea)
        network_arch_mat = network_layer_to_space(network_path_mat)

        self.maxdisp = maxdisp
        self.feature = newFeature(network_arch_fea, cell_arch_fea)
        self.matching = newMatching(network_arch_mat, cell_arch_mat)
        self.cost_interpolation = CostInterpolation(self.maxdisp)
        self.maxdisp_downsampleing = maxdisp_downsampleing

    def forward(self, x, y):
        x = self.feature(x)
        y = self.feature(y)

        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1] * 2, int(self.maxdisp / self.maxdisp_downsampleing),
                                   x.size()[2], x.size()[3]).zero_()
        for i in range(int(self.maxdisp / self.maxdisp_downsampleing)):
            if i > 0:
                cost[:, :x.size()[1], i, :, i:] = x[:, :, :, i:]
                cost[:, x.size()[1]:, i, :, i:] = y[:, :, :, :-i]
            else:
                cost[:, :x.size()[1], i, :, i:] = x
                cost[:, x.size()[1]:, i, :, i:] = y

        cost = self.matching(cost)
        cost = self.cost_interpolation(cost)
        return cost

    def get_params(self):
        back_bn_params, back_no_bn_params = self.encoder.get_params()
        tune_wd_params = list(self.aspp.parameters()) \
                         + list(self.decoder.parameters()) \
                         + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params
