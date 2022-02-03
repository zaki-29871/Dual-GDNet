from GDNet.module import *

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2))

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)

        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)

        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)

        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2),
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv4 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv5 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv21 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv22 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv31 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv32 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv41 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv42 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_gd1 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd2 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd3 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd4 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd5 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_gd11 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd12 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd21 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd22 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd31 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd32 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd41 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd42 = nn.Conv2d(48, 2880, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg3 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg4 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

    def forward(self, x):
        x = self.conv0(x)  # H, W
        rem = x

        # gd1, gd2, gd3, gd4, gd5: 1920, H/8, W/8
        # 1920 = 32*6*10
        x = self.conv1(x)
        gd1 = self.weight_gd1(x)
        x = self.conv2(x)
        gd2 = self.weight_gd2(x)
        x = self.conv3(x)
        gd3 = self.weight_gd3(x)
        x = self.conv4(x)
        gd4 = self.weight_gd4(x)
        x = self.conv5(x)
        gd5 = self.weight_gd5(x)

        # gd11, gd12, gd21, gd22: 2880, H/16, W/16
        # 2880 = 48*6*10
        x = self.conv11(x)
        gd11 = self.weight_gd11(x)
        x = self.conv12(x)
        gd12 = self.weight_gd12(x)
        x = self.conv21(x)
        gd21 = self.weight_gd21(x)
        x = self.conv22(x)
        gd22 = self.weight_gd22(x)
        x = self.conv31(x)
        gd31 = self.weight_gd31(x)
        x = self.conv32(x)
        gd32 = self.weight_gd32(x)
        x = self.conv41(x)
        gd41 = self.weight_gd41(x)
        x = self.conv42(x)
        gd42 = self.weight_gd42(x)

        # lgx: 75, H, W
        # 75 = 3*5*5
        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)
        lg3 = self.weight_lg3(rem)
        lg4 = self.weight_lg4(rem)

        return dict([
            ('gd1', gd1),
            ('gd2', gd2),
            ('gd3', gd3),
            ('gd4', gd4),
            ('gd5', gd5),
            ('gd11', gd11),
            ('gd12', gd12),
            ('gd21', gd21),
            ('gd22', gd22),
            ('gd31', gd31),
            ('gd32', gd32),
            ('gd41', gd41),
            ('gd42', gd42),
            ('lg1', lg1),
            ('lg2', lg2),
            ('lg3', lg3),
            ('lg4', lg4)])

class CostInterpolate(nn.Module):

    def __init__(self, max_disparity=192):
        super(CostInterpolate, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), scale_factor=8, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        return x

class CostInterpolateAggregation(nn.Module):
    def __init__(self, max_disparity=192):
        super(CostInterpolateAggregation, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.lga = LGA(5)

    def forward(self, x, lg1, lg2, lg3, lg4):
        x = F.interpolate(self.conv32x1(x), scale_factor=8, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        x = self.lga(x, lg1)     # D, H, W
        x = F.leaky_relu(x)
        x = self.lga(x, lg2)     # D, H, W
        x = F.leaky_relu(x)
        x = self.lga(x, lg3)     # D, H, W
        x = F.leaky_relu(x)
        x = self.lga(x, lg4)     # D, H, W
        x = F.leaky_relu(x)
        return x

class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.conv1b = Conv2x(32, 48, is_3d=True)
        self.conv2b = Conv2x(48, 64, is_3d=True)
        self.deconv2b = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1b = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.conv1c = Conv2x(32, 48, is_3d=True)
        self.conv2c = Conv2x(48, 64, is_3d=True)
        self.deconv2c = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1c = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.conv1d = Conv2x(32, 48, is_3d=True)
        self.conv2d = Conv2x(48, 64, is_3d=True)
        self.deconv2d = Conv2x(64, 48, deconv=True, is_3d=True)
        self.deconv1d = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)

        self.gd1 = GD6_Block(32, 3)
        self.gd2 = GD6_Block(32, 3)
        self.gd3 = GD6_Block(32, 3)
        self.gd4 = GD6_Block(32, 3)
        self.gd5 = GD6_Block(32, 3)
        self.gd11 = GD6_Block(48, 3)
        self.gd12 = GD6_Block(48, 3)
        self.gd21 = GD6_Block(48, 3)
        self.gd22 = GD6_Block(48, 3)
        self.gd31 = GD6_Block(48, 3)
        self.gd32 = GD6_Block(48, 3)
        self.gd41 = GD6_Block(48, 3)
        self.gd42 = GD6_Block(48, 3)

        self.cost0 = CostInterpolate(self.maxdisp)
        self.cost1 = CostInterpolate(self.maxdisp)
        self.cost2 = CostInterpolate(self.maxdisp)
        self.cost3 = CostInterpolate(self.maxdisp)
        self.cost4 = CostInterpolateAggregation(self.maxdisp)

    def forward(self, x, g):
        x = self.conv_start(x)

        # Part 1
        # x: 32, D/8, H/8, W/8
        # gd1: 1920, H/8, W/8
        # gd11, gd12: 2880, H/16, W/16
        x = self.gd1(x, g['gd1'])
        rem0 = x

        if self.training:
            cost0 = self.cost0(x)

        x = self.conv1a(x)
        x = self.gd11(x, g['gd11'])
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.gd12(x, g['gd12'])
        rem1 = x
        x = self.deconv1a(x, rem0)

        # Part 2
        # x: 32, D/8, H/8, W/8
        # gd2: 1920, H/8, W/8
        # gd21, gd22: 2880, H/16, W/16
        x = self.gd2(x, g['gd2'])
        rem0 = x
        if self.training:
            cost1 = self.cost1(x)

        x = self.conv1b(x, rem1)
        x = self.gd21(x, g['gd21'])
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.deconv2b(x, rem1)
        x = self.gd22(x, g['gd22'])
        rem1 = x
        x = self.deconv1b(x, rem0)

        # Part 3
        # x: 32, D/8, H/8, W/8
        # gd3: 1920, H/8, W/8
        # gd31, gd32: 2880, H/16, W/16
        x = self.gd3(x, g['gd3'])
        rem0 = x
        if self.training:
            cost2 = self.cost2(x)

        x = self.conv1c(x, rem1)
        x = self.gd31(x, g['gd31'])
        rem1 = x
        x = self.conv2c(x, rem2)
        rem2 = x
        x = self.deconv2c(x, rem1)
        x = self.gd32(x, g['gd32'])
        rem1 = x
        x = self.deconv1c(x, rem0)

        # Part 4
        # x: 32, D/8, H/8, W/8
        # gd4: 1920, H/8, W/8
        # gd41, gd42: 2880, H/16, W/16
        x = self.gd4(x, g['gd4'])
        rem0 = x
        if self.training:
            cost3 = self.cost3(x)

        x = self.conv1d(x, rem1)
        x = self.gd41(x, g['gd41'])
        rem1 = x
        x = self.conv2d(x, rem2)
        x = self.deconv2d(x, rem1)
        x = self.gd42(x, g['gd42'])
        x = self.deconv1d(x, rem0)

        # Part 5
        # gd5: 1920, H/4, W/4
        x = self.gd5(x, g['gd5'])

        # x: 32, D/4, H/4, W/4
        # lg1, lg2, lg3, lg4: 75, H, W
        cost4 = self.cost4(x, g['lg1'], g['lg2'], g['lg3'], g['lg4'])
        if self.training:
            return cost0, cost1, cost2, cost3, cost4
        else:
            return cost4

class GDNet_sdc6(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDNet_sdc6, self).__init__()
        self.max_disparity = max_disparity

        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = nn.Sequential(BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
                                    BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv_y = nn.Sequential(BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
                                    BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.guidance = Guidance()
        self.feature = Feature()
        self.cost_volume = CostVolume(max_disparity/8)
        self.cost_aggregation = CostAggregation(self.max_disparity)
        self.flip = False

    def forward(self, x, y):
        if self.flip:
            flip_x = x.data.cpu().numpy()
            flip_y = y.data.cpu().numpy()
            flip_x = torch.tensor(flip_x[..., ::-1].copy()).cuda()
            flip_y = torch.tensor(flip_y[..., ::-1].copy()).cuda()
            x = flip_y
            y = flip_x

        g = self.conv_start(x)  # 32, H, W
        x = self.feature(x)
        y = self.feature(y)

        rem = x  # 32, H/4, W/4
        x = self.conv_x(x)  # 32, H/8, W/8
        y = self.conv_y(y)  # 32, H/8, W/8

        x = self.cost_volume(x, y)  # 64, D/8, H/8, W/8

        x1 = self.conv_refine(rem)  # 32, H/4, W/4

        # 32, H, W
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)

        g = torch.cat((g, x1), 1)  # 64, H, W

        # gd1, gd2, gd3: 1920, H/8, W/8
        # 1920 = 32*6*10
        # gd11, gd12, gd13, gd14: 2880, H/16, W/16
        # 2880 = 48*6*10
        # lg1, lg2, lg3, lg4: 75, H, W
        g = self.guidance(g)

        return self.cost_aggregation(x, g)