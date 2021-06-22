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
            BasicConv(32, 32, kernel_size=5, stride=2, padding=2))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv3 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv13 = BasicConv(48, 48, kernel_size=3, padding=1)
        self.conv14 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_gd1 = nn.Conv2d(32, 1280, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd2 = nn.Conv2d(32, 1280, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd3 = nn.Conv2d(32, 1280, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_gd11 = nn.Conv2d(48, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd12 = nn.Conv2d(48, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd13 = nn.Conv2d(48, 1920, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_gd14 = nn.Conv2d(48, 1920, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

    def forward(self, x):
        x = self.conv0(x)  # H, W
        rem = x

        # gd1, gd2, gd3: 1280, H/4, W/4
        # 1280 = 32*4*10
        x = self.conv1(x)
        gd1 = self.weight_gd1(x)
        x = self.conv2(x)
        gd2 = self.weight_gd2(x)
        x = self.conv3(x)
        gd3 = self.weight_gd3(x)

        # gd11, gd12, gd13, gd14: 1920, H/8, W/8
        # 1920 = 48*4*10
        x = self.conv11(x)
        gd11 = self.weight_gd11(x)
        x = self.conv12(x)
        gd12 = self.weight_gd12(x)
        x = self.conv13(x)
        gd13 = self.weight_gd13(x)
        x = self.conv14(x)
        gd14 = self.weight_gd14(x)

        # lg1: 75, H, W
        # 75 = 3*5*5
        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)

        return dict([
            ('gd1', gd1),
            ('gd2', gd2),
            ('gd3', gd3),
            ('gd11', gd11),
            ('gd12', gd12),
            ('gd13', gd13),
            ('gd14', gd14),
            ('lg1', lg1),
            ('lg2', lg2)])

class CostInterpolate(nn.Module):

    def __init__(self, max_disparity=192):
        super(CostInterpolate, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), scale_factor=4, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        return x

class CostInterpolateAggregation(nn.Module):
    def __init__(self, max_disparity=192):
        super(CostInterpolateAggregation, self).__init__()
        self.max_disparity = max_disparity
        self.conv32x1 = nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.lga = LGA(5)

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), scale_factor=4, mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)  # D, H, W
        x = self.lga(x, lg1)     # D, H, W
        x = F.leaky_relu(x)
        x = self.lga(x, lg2)     # D, H, W
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

        self.gd1 = GD4_Block(32, 3)
        self.gd2 = GD4_Block(32, 3)
        self.gd3 = GD4_Block(32, 3)
        self.gd11 = GD4_Block(48, 3)
        self.gd12 = GD4_Block(48, 3)
        self.gd13 = GD4_Block(48, 3)
        self.gd14 = GD4_Block(48, 3)

        self.cost0 = CostInterpolate(self.maxdisp)
        self.cost1 = CostInterpolate(self.maxdisp)
        self.cost2 = CostInterpolateAggregation(self.maxdisp)

    def forward(self, x, g):
        x = self.conv_start(x)

        # Part 1
        # x: 32, D/4, H/4, W/4
        # gd1: 1280, H/4, W/4
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
        x = self.gd2(x, g['gd2'])
        rem0 = x
        if self.training:
            cost1 = self.cost1(x)

        x = self.conv1b(x, rem1)
        x = self.gd13(x, g['gd13'])
        rem1 = x
        x = self.conv2b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.gd14(x, g['gd14'])
        x = self.deconv1b(x, rem0)

        # Part 3
        x = self.gd3(x, g['gd3'])

        # x: 32, D/4, H/4, W/4
        # lg1: 75, H, W
        # lg2: 75, H, W
        cost2 = self.cost2(x, g['lg1'], g['lg2'])
        if self.training:
            return cost0, cost1, cost2
        else:
            return cost2

class GDNet_mdc4(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDNet_mdc4, self).__init__()
        self.max_disparity = max_disparity

        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.guidance = Guidance()
        self.feature = Feature()
        self.cost_volume = CostVolume(max_disparity/4)
        self.cost_aggregation = CostAggregation(self.max_disparity)

    def forward(self, x, y):
        g = self.conv_start(x)  # 32, H, W
        x = self.feature(x)
        y = self.feature(y)

        rem = x
        x = self.conv_x(x)  # 32, H/4, W/4
        y = self.conv_y(y)  # 32, H/4, W/4

        x = self.cost_volume(x, y)  # 64, D/4, H/4, W/4

        x1 = self.conv_refine(rem)  # 32, H/4, W/4

        # 32, H, W
        x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)

        g = torch.cat((g, x1), 1)  # 64, H, W

        # gd1, gd2, gd3: 1280, H/4, W/4
        # 1280 = 32*4*10
        # gd11, gd12, gd13, gd14: 1920, H/8, W/8
        # 1920 = 48*4*10
        # lg1: 75, H, W
        # lg2: 75, H, W
        g = self.guidance(g)

        return self.cost_aggregation(x, g)