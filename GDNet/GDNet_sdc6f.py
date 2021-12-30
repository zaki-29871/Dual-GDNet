from GDNet.function import *
import GDNet.GDNet_sdc6


class GDNet_sdc6f(nn.Module):
    def __init__(self, max_disparity=192):
        super(GDNet_sdc6f, self).__init__()
        self.max_disparity = max_disparity
        self.model = GDNet.GDNet_sdc6.GDNet_sdc6(max_disparity)
        self.flip = False

    def forward(self, x, y):
        self.model.flip = self.flip

        if self.training:
            if self.flip:
                cost0, cost1, cost2, cost3, cost4 = self.model(x, y)
                cost0f = FlipCost.apply(cost0)[..., self.max_disparity:]
                cost1f = FlipCost.apply(cost2)[..., self.max_disparity:]
                cost2f = FlipCost.apply(cost2)[..., self.max_disparity:]
                cost3f = FlipCost.apply(cost3)[..., self.max_disparity:]
                cost4f = FlipCost.apply(cost4)[..., self.max_disparity:]
                return cost0f, cost1f, cost2f, cost3f, cost4f

            else:
                cost0, cost1, cost2, cost3, cost4 = self.model(x, y)
                return cost0, cost1, cost2, cost3, cost4
        else:
            if self.flip:
                cost = self.model(x, y)
                # cost = FlipCost.apply(cost)
                return cost

            else:
                cost = self.model(x, y)
                return cost
