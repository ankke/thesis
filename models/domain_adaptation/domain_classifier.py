import torch.nn as nn
from .functional import ReverseLayerF
import torch


class Discriminator(nn.Module):
    """
    A 2-layer MLP for domain classification based on DANN.
    """

    def __init__(self, in_size=43008, h=2048, out_size=1):
        """
        Arguments:
            in_size: size of the input
            h: hidden layer size
            out_size: size of the output
        """

        super().__init__()
        self.h = h
        self.net = nn.Sequential(
            nn.Linear(in_size, h),
            nn.BatchNorm1d(h),
            nn.ReLU(True),
            nn.Linear(h, 2),
            nn.ReLU(),
            nn.LogSoftmax()
        )
        self.out_size = out_size

    def forward(self, srcs, alpha):
        """"""
        features = []
        for feature in srcs[1:]:
            feature = torch.flatten(feature, start_dim=1)
            features.append(feature)
        conc_features = torch.concat(features, dim=1)
        reverse_features = ReverseLayerF.apply(conc_features, alpha)
        domain_output = self.net(reverse_features)
        return domain_output