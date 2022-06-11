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

        self.conv1c = Conv2x(32, 48)
        self.conv2c = Conv2x(48, 64)
        self.conv3c = Conv2x(64, 96)

        self.deconv3c = Conv2x(96, 64, deconv=True)
        self.deconv2c = Conv2x(64, 48, deconv=True)
        self.deconv1c = Conv2x(48, 32, deconv=True)

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
        rem2 = x
        x = self.deconv2b(x, rem1)
        rem1 = x
        x = self.deconv1b(x, rem0)

        x = self.conv1c(x, rem1)
        rem1 = x
        x = self.conv2c(x, rem2)
        rem2 = x
        x = self.conv3c(x, rem3)

        x = self.deconv3c(x, rem2)
        x = self.deconv2c(x, rem1)
        x = self.deconv1c(x, rem0)

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
        self.conv6 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv7 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv8 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv9 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.weight_gd1 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd2 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd3 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd4 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd5 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd6 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd7 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd8 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd9 = nn.Conv2d(32, 1920, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg3 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg4 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg5 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg6 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg7 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg8 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg9 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
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
        x = self.conv6(x)
        gd6 = self.weight_gd6(x)
        x = self.conv7(x)
        gd7 = self.weight_gd7(x)
        x = self.conv8(x)
        gd8 = self.weight_gd8(x)
        x = self.conv9(x)
        gd9 = self.weight_gd9(x)

        # lgx: 75, H, W
        # 75 = 3*5*5
        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)
        lg3 = self.weight_lg3(rem)
        lg4 = self.weight_lg4(rem)
        lg5 = self.weight_lg5(rem)
        lg6 = self.weight_lg6(rem)
        lg7 = self.weight_lg7(rem)
        lg8 = self.weight_lg8(rem)
        lg9 = self.weight_lg9(rem)

        return dict([
            ('gd1', gd1),
            ('gd2', gd2),
            ('gd3', gd3),
            ('gd4', gd4),
            ('gd5', gd5),
            ('gd6', gd6),
            ('gd7', gd7),
            ('gd8', gd8),
            ('gd9', gd9),
            ('lg1', lg1),
            ('lg2', lg2),
            ('lg3', lg3),
            ('lg4', lg4),
            ('lg5', lg5),
            ('lg6', lg6),
            ('lg7', lg7),
            ('lg8', lg8),
            ('lg9', lg9)])


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

    def forward(self, x, lga_list: list):
        x = F.interpolate(self.conv32x1(x), scale_factor=8, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        for lga in lga_list[:-1]:
            x = self.lga(x, lga)  # D, H, W
            x = F.leaky_relu(x)
        x = self.lga(x, lga_list[-1])  # D, H, W
        x = F.softmax(x, dim=1)
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
        self.gd6 = GD6_Block(32, 3)
        self.gd7 = GD6_Block(32, 3)
        self.gd8 = GD6_Block(32, 3)
        self.gd9 = GD6_Block(32, 3)

        self.conv11 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv12 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv21 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv22 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv31 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv32 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv41 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv42 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv51 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv52 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv61 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv62 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv71 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv72 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv81 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))
        self.conv82 = nn.Sequential(BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True),
                                    BasicConv(48, 48, kernel_size=3, stride=1, padding=1, is_3d=True))

        self.cost0 = CostInterpolate(self.maxdisp)
        self.cost1 = CostInterpolate(self.maxdisp)
        self.cost2 = CostInterpolateAggregation(self.maxdisp)

    def forward(self, x, g):
        x = self.conv_start(x)

        # Part 1
        # x: 32, D/8, H/8, W/8
        # gd1: 1920, H/8, W/8
        x = self.gd1(x, g['gd1'])
        rem0 = x

        if self.training:
            cost0 = self.cost0(x)

        x = self.conv1a(x)
        x = self.conv11(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.deconv2a(x, rem1)
        x = self.conv12(x)
        rem1 = x
        x = self.deconv1a(x, rem0)

        # Part 2
        # x: 32, D/8, H/8, W/8
        # gd2: 1920, H/8, W/8
        x = self.gd2(x, g['gd2'])
        rem0 = x

        x = self.conv1b(x, rem1)
        x = self.conv21(x)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.deconv2b(x, rem1)
        x = self.conv22(x)
        rem1 = x
        x = self.deconv1b(x, rem0)

        # Part 3
        # x: 32, D/8, H/8, W/8
        # gd3: 1920, H/8, W/8
        x = self.gd3(x, g['gd3'])
        rem0 = x

        x = self.conv1c(x, rem1)
        x = self.conv31(x)
        rem1 = x
        x = self.conv2c(x, rem2)
        rem2 = x
        x = self.deconv2c(x, rem1)
        x = self.conv32(x)
        rem1 = x
        x = self.deconv1c(x, rem0)

        # Part 4
        # x: 32, D/8, H/8, W/8
        # gd4: 1920, H/8, W/8
        x = self.gd4(x, g['gd4'])
        rem0 = x

        x = self.conv1d(x, rem1)
        x = self.conv41(x)
        rem1 = x
        x = self.conv2d(x, rem2)
        rem2 = x
        x = self.deconv2d(x, rem1)
        x = self.conv42(x)
        rem1 = x
        x = self.deconv1d(x, rem0)

        # Part 5
        # x: 32, D/8, H/8, W/8
        # gd5: 1920, H/8, W/8
        x = self.gd5(x, g['gd5'])
        rem0 = x

        if self.training:
            cost1 = self.cost1(x)

        x = self.conv1d(x, rem1)
        x = self.conv51(x)
        rem1 = x
        x = self.conv2d(x, rem2)
        rem2 = x
        x = self.deconv2d(x, rem1)
        x = self.conv52(x)
        rem1 = x
        x = self.deconv1d(x, rem0)

        # Part 6
        # x: 32, D/8, H/8, W/8
        # gd6: 1920, H/8, W/8
        x = self.gd6(x, g['gd6'])
        rem0 = x

        x = self.conv1d(x, rem1)
        x = self.conv61(x)
        rem1 = x
        x = self.conv2d(x, rem2)
        rem2 = x
        x = self.deconv2d(x, rem1)
        x = self.conv62(x)
        rem1 = x
        x = self.deconv1d(x, rem0)

        # Part 7
        # x: 32, D/8, H/8, W/8
        # gd7: 1920, H/8, W/8
        x = self.gd7(x, g['gd7'])
        rem0 = x

        x = self.conv1d(x, rem1)
        x = self.conv71(x)
        rem1 = x
        x = self.conv2d(x, rem2)
        rem2 = x
        x = self.deconv2d(x, rem1)
        x = self.conv72(x)
        rem1 = x
        x = self.deconv1d(x, rem0)

        # Part 8
        # x: 32, D/8, H/8, W/8
        # gd8: 1920, H/8, W/8
        x = self.gd8(x, g['gd8'])
        rem0 = x

        x = self.conv1d(x, rem1)
        x = self.conv81(x)
        rem1 = x
        x = self.conv2d(x, rem2)
        x = self.deconv2d(x, rem1)
        x = self.conv82(x)
        x = self.deconv1d(x, rem0)

        # Part 9
        # x: 32, D/8, H/8, W/8
        # gd8: 1920, H/8, W/8
        x = self.gd9(x, g['gd9'])

        # x: 32, D/4, H/4, W/4
        # lg1, lg2, lg3, lg4: 75, H, W
        lga_list = [g['lg1'], g['lg2'], g['lg3'], g['lg4'], g['lg5'], g['lg6'], g['lg7'], g['lg8'], g['lg9']]
        cost2 = self.cost2(x, lga_list)
        if self.training:
            return cost0, cost1, cost2
        else:
            return cost2


class GDNet_sd9d6(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDNet_sd9d6, self).__init__()
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
        self.cost_volume = CostVolume(max_disparity / 8)
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
        # lg1, lg2, lg3, lg4: 75, H, W
        g = self.guidance(g)

        return self.cost_aggregation(x, g)
