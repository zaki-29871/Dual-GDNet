from GDNet.function import *
import GDNet.GDNet_sd9c6


class GDNet_sd9c6f(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDNet_sd9c6f, self).__init__()
        self.max_disparity = max_disparity
        self.model = GDNet.GDNet_sd9c6.GDNet_sd9c6(max_disparity)
        self.flip = False

    def forward(self, x, y):
        self.model.flip = self.flip

        if self.training:
            if self.flip:
                cost0, cost1, cost2 = self.model(x, y)
                cost0f = FlipCost.apply(cost0)[..., self.max_disparity:]
                cost1f = FlipCost.apply(cost2)[..., self.max_disparity:]
                cost2f = FlipCost.apply(cost2)[..., self.max_disparity:]
                return cost0f, cost1f, cost2f

            else:
                cost0, cost1, cost2 = self.model(x, y)
                return cost0, cost1, cost2
        else:
            if self.flip:
                cost = self.model(x, y)
                # cost = FlipCost.apply(cost)
                return cost

            else:
                cost = self.model(x, y)
                return cost
