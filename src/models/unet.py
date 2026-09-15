import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for binary image segmentation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            base_channels,
        )

        self.enc2 = DoubleConv(
            base_channels,
            base_channels * 2,
        )

        self.enc3 = DoubleConv(
            base_channels * 2,
            base_channels * 4,
        )

        self.enc4 = DoubleConv(
            base_channels * 4,
            base_channels * 8,
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            base_channels * 8,
            base_channels * 16,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2,
        )

        self.dec4 = DoubleConv(
            base_channels * 16,
            base_channels * 8,
        )

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            base_channels * 8,
            base_channels * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            base_channels * 2,
            base_channels,
        )

        # Output layer
        self.out = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

def get_activation(name: str) -> nn.Module:
    """Return the activation function specified by name."""

    activations = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True,
        ),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}"
        )

    return activations[name]

class ConvBlock(nn.Module):
    """
    Convolutional block used in the U-Net encoder and decoder.

    Each block applies a configurable number of convolutional layers,
    followed by BatchNorm and a configurable activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        kernel_size: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        if num_convs < 1:
            raise ValueError(
                "num_convs must be at least 1."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd."
            )

        padding = kernel_size // 2

        layers = []

        for conv_index in range(num_convs):
            conv_in_channels = (
                in_channels
                if conv_index == 0
                else out_channels
            )

            layers.extend(
                [
                    nn.Conv2d(
                        conv_in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    get_activation(activation),
                ]
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConfigurableUNet(nn.Module):
    """
    Configurable U-Net for binary image segmentation.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    base_channels:
        Number of channels in the first encoder block.
        Channels are doubled after each downsampling step.
    depth:
        Number of encoder/decoder levels.
    num_convs:
        Number of convolutional layers in each block.
    kernel_size:
        Spatial size of convolutional kernels.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        num_convs: int = 2,
        kernel_size: int = 3,
        activation: str = "relu",
    ):
        super().__init__()

        if base_channels < 1:
            raise ValueError("base_channels must be at least 1.")

        if depth < 1:
            raise ValueError("depth must be at least 1.")

        if num_convs < 1:
            raise ValueError("num_convs must be at least 1.")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.num_convs = num_convs
        self.kernel_size = kernel_size

        # Channel dimensions at each encoder level.
        channels = [
            base_channels * (2 ** level)
            for level in range(depth)
        ]

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------

        self.encoder_blocks = nn.ModuleList()

        for level, out_channels_level in enumerate(channels):
            in_channels_level = (
                in_channels
                if level == 0
                else channels[level - 1]
            )

            self.encoder_blocks.append(
                ConvBlock(
                    in_channels=in_channels_level,
                    out_channels=out_channels_level,
                    num_convs=num_convs,
                    kernel_size=kernel_size,
                )
            )

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------

        bottleneck_channels = channels[-1] * 2

        self.bottleneck = ConvBlock(
            in_channels=channels[-1],
            out_channels=bottleneck_channels,
            num_convs=num_convs,
            kernel_size=kernel_size,
        )

        # ------------------------------------------------------------------
        # Downsampling
        # ------------------------------------------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        current_channels = bottleneck_channels

        for level in reversed(range(depth)):
            out_channels_level = channels[level]

            self.upconvs.append(
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels_level,
                    kernel_size=2,
                    stride=2,
                )
            )

            self.decoder_blocks.append(
                ConvBlock(
                    in_channels=out_channels_level * 2,
                    out_channels=out_channels_level,
                    num_convs=num_convs,
                    kernel_size=kernel_size,
                )
            )

            current_channels = out_channels_level

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------

        self.out = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        encoder_features = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upconv, decoder_block, skip in zip(
            self.upconvs,
            self.decoder_blocks,
            reversed(encoder_features),
        ):
            x = upconv(x)

            x = torch.cat(
                [x, skip],
                dim=1,
            )

            x = decoder_block(x)

        return self.out(x)