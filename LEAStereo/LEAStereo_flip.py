from GDNet.function import *
import LEAStereo.LEAStereo


class LEAStereo_flip(nn.Module):
    def __init__(self, max_disparity, maxdisp_downsampleing):
        super(LEAStereo_flip, self).__init__()
        self.max_disparity = max_disparity
        self.model = LEAStereo.LEAStereo.LEAStereo(max_disparity, maxdisp_downsampleing)
        self.flip = False

    def forward(self, x, y):
        self.model.flip = self.flip

        if self.training:
            if self.flip:
                cost = self.model(x, y)
                costf = FlipCost.apply(cost)[..., self.max_disparity:]
                return costf

            else:
                cost = self.model(x, y)
                return cost
        else:
            if self.flip:
                cost = self.model(x, y)
                # cost = FlipCost.apply(cost)
                return cost

            else:
                cost = self.model(x, y)
                return cost
