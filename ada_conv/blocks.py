import torch
from torch import nn
from torch.nn import functional as F


class GlobalStyleEncoder(nn.Module):
    def __init__(
        self, style_feat_shape: tuple[int], style_descriptor_shape: tuple[int]
    ) -> None:
        super().__init__()
        self.style_feat_shape = style_feat_shape
        self.style_descriptor_shape = style_descriptor_shape
        channels = self.style_feat_shape[0]

        self.style_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            # Block 2
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            # Block 3
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
        )

        in_features = int(
            self.style_feat_shape[0]
            * (self.style_feat_shape[1] // 8)
            * (self.style_feat_shape[2] // 8)
        )
        out_features = int(
            self.style_descriptor_shape[0]
            * self.style_descriptor_shape[1]
            * self.style_descriptor_shape[2]
        )
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # style encoder
        x = self.style_encoder(x)
        # fully connected
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        # global embeddings
        w = w.view(
            -1,
            self.style_descriptor_shape[0],
            self.style_descriptor_shape[1],
            self.style_descriptor_shape[2],
        )
        return w


class KernelPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        style_kernel: int,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.style_kernel = style_kernel

        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=self.out_channels * (self.in_channels // self.groups),
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels * (self.out_channels // self.groups),
                kernel_size=(1, 1),
            ),
        )

        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels,
                kernel_size=(1, 1),
            ),
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor]:
        # depthwise kernel
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.view(
            -1,
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        # pointwise kernel
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.view(
            -1,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        # bias
        bias = self.bias_predictor(w)
        bias = bias.view(-1, self.out_channels)
        return (dw_kernel, pw_kernel, bias)


class AdaConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        dw_kernels: torch.Tensor,
        pw_kernels: torch.Tensor,
        biases: torch.Tensor,
    ) -> torch.Tensor:
        x = F.instance_norm(x)

        out = []
        for i in range(x.shape[0]):
            y = self._depthwise_separable_conv2D(
                x[i : i + 1], dw_kernels[i], pw_kernels[i], biases[i]
            )
            out.append(y)
        out = torch.cat(out, dim=0)

        out = self.spatial_conv(out)
        return out

    def _depthwise_separable_conv2D(
        self,
        x: torch.Tensor,
        dw_kernel: torch.Tensor,
        w_pointwise: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        padding = (dw_kernel.size(-1) - 1) // 2
        pad = (padding, padding, padding, padding)
        x = F.pad(x, pad=pad, mode="reflect")
        x = F.conv2d(x, dw_kernel, groups=self.groups)
        x = F.conv2d(x, w_pointwise, groups=self.groups, bias=bias)
        return x
