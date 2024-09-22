import torch
from torch import nn
from torchvision import models
from torchvision.transforms import transforms


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_factor = 8
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        vgg19_feats = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        blocks = list()
        blocks.append(nn.Sequential(*vgg19_feats[:2]))  # input -> relu1_1
        blocks.append(nn.Sequential(*vgg19_feats[2:7]))  # relu1_1 -> relu2_1
        blocks.append(nn.Sequential(*vgg19_feats[7:12]))  # relu2_1 -> relu3_1
        blocks.append(nn.Sequential(*vgg19_feats[12:21]))  # relu3_1 -> relu4_1

        self.blocks = nn.ModuleList(blocks)

        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    layer.padding_mode = "reflect"
                elif isinstance(layer, nn.ReLU):
                    layer.inplace = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalize(x)

        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)

        return feats

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
