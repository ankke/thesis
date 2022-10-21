import torch
from torch import nn
from models.deformable_detr_backbone import Backbone
from models.utils import nested_tensor_from_tensor_list, NestedTensor


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        return_interm_layers = config.MODEL.ENCODER.MASKS or (
                config.MODEL.ENCODER.NUM_FEATURE_LEVELS > 1)
        self.encoder = Backbone(
            config.MODEL.ENCODER.BACKBONE,
            config.MODEL.ENCODER.BACKBONE_INIT_WEIGHTS,
            True,
            return_interm_layers,
            config.MODEL.ENCODER.DILATION
        )

    def forward(self, samples):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        x = self.encoder(samples)
        for i in range(3):
            features, mask = x[f"{i}"].decompose()
        feature, mask = x["2"].decompose()
        print(feature.shape)
        return feature
