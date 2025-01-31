# Third Party
import torch as t
import numpy as np

# Internal
from . import enums

conv_dict = {1: t.nn.Conv1d, 2: t.nn.Conv2d, 3: t.nn.Conv3d}
conv_t_dict = {
    1: t.nn.ConvTranspose1d,
    2: t.nn.ConvTranspose2d,
    3: t.nn.ConvTranspose3d,
}
batch_norm_dict = {1: t.nn.BatchNorm1d, 2: t.nn.BatchNorm2d, 3: t.nn.BatchNorm3d}
max_pool_dict = {1: t.nn.MaxPool1d, 2: t.nn.MaxPool2d, 3: t.nn.MaxPool3d}


class CvNmAc(t.nn.Module):
    """A layer containing one or more convolutions, a normalization, and an activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: enums.NormalizationEnum = enums.NormalizationEnum.group,
        channels_per_group: int = 8,
        convolutions: int = 2,
        nonlinearity: enums.NonlinearityEnum = enums.NonlinearityEnum.relu,
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding.
            norm: Normalization strategy.
            channels_per_group: For use with Group Normalization.
            convolutions: Number of convolutions.
        """
        super(CvNmAc, self).__init__()
        modules = t.nn.ModuleList()

        # Convolution
        for c in range(convolutions):
            modules.append(
                conv_dict[dimension](
                    in_channels if c == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                )
            )

        # Normalization
        match norm:
            case enums.NormalizationEnum.group:
                modules.append(
                    t.nn.GroupNorm(out_channels // channels_per_group, out_channels)
                )
            case enums.NormalizationEnum.instance:
                modules.append(t.nn.GroupNorm(out_channels, out_channels))
            case enums.NormalizationEnum.layer:
                modules.append(t.nn.GroupNorm(1, out_channels))
            case enums.NormalizationEnum.batch:
                modules.append(batch_norm_dict[dimension](out_channels))
            case _:
                assert False, f"Option '{norm}' not recognized."

        # Activation
        match nonlinearity:
            case enums.NonlinearityEnum.relu:
                modules.append(t.nn.ReLU(inplace=True))
            case enums.NonlinearityEnum.leaky_relu:
                modules.append(t.nn.LeakyReLU(inplace=True))
            case enums.NonlinearityEnum.none:
                pass
            case _:
                assert False, f"Nonlinearity {nonlinearity} not recognized."

        self.conv = t.nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class CvNmAcxN(t.nn.Module):
    """A layer which repeats `CvNmAc` a specified number of times."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: enums.NormalizationEnum = enums.NormalizationEnum.group,
        channels_per_group: int = 8,
        convolutions: int = 2,
        nonlinearity: enums.NonlinearityEnum = enums.NonlinearityEnum.relu,
        repetitions: int = 2,
        residual: bool = False,
    ):
        """Initialization.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels in the output.
            dimension: Dimensionality of the input.
            kernel_size: Size of the convolution kernel.
            stride: Convolution stride.
            padding: Padding.
            norm: Normalization strategy.
            channels_per_group: For use with Group Normalization.
            convolutions: Number of convolutions.
            repetitions: Number of times to repeat `CvNmAc`.
        """
        super(CvNmAcxN, self).__init__()
        self.residual = residual
        self.nonlinearity = nonlinearity
        modules = t.nn.ModuleList()

        for r in range(repetitions):
            if residual and r == repetitions - 1:
                nl = enums.NonlinearityEnum.none
            else:
                nl = nonlinearity
            modules.append(
                CvNmAc(
                    in_channels if r == 0 else out_channels,
                    out_channels,
                    dimension,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    norm=norm,
                    channels_per_group=channels_per_group,
                    convolutions=convolutions,
                    nonlinearity=nl,
                )
            )

        self.conv = t.nn.Sequential(*modules)
        if self.residual:
            self.conv_residual = CvNmAc(
                in_channels,
                out_channels,
                dimension,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                channels_per_group=channels_per_group,
                convolutions=1,
                nonlinearity=enums.NonlinearityEnum.none,
            )

    def forward(self, x):
        if self.residual:
            x = self.conv(x) + self.conv_residual(x)
            match self.nonlinearity:
                case enums.NonlinearityEnum.relu:
                    x = t.nn.ReLU(inplace=True)(x)
                case enums.NonlinearityEnum.leaky_relu:
                    x = t.nn.LeakyReLU(inplace=True)(x)
                case enums.NonlinearityEnum.none:
                    pass
                case _:
                    assert False, f"Nonlinearity {nonlinearity} not recognized."
            return x
        else:
            return self.conv(x)


class Attn(t.nn.Module):
    """
    https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self,
        F_g: int,  # Number of features in most recent layer
        F_l: int,  # Number of features in the skip connection layer
        F_int: int,  # Number of attention coefficients
        dimension: int,
        channels_per_group: int = 8,
    ):
        super(Attn, self).__init__()
        conv = conv_dict[dimension]

        self.W_g = t.nn.Sequential(
            conv(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            t.nn.GroupNorm(F_int // channels_per_group, F_int),
        )

        self.W_x = t.nn.Sequential(
            conv(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            t.nn.GroupNorm(F_int // channels_per_group, F_int),
        )

        self.psi = t.nn.Sequential(
            conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            t.nn.GroupNorm(1, 1),
            t.nn.Sigmoid(),
        )

        self.activation = t.nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


