"""Encoder-Decoder 2D U-Net implementation."""

import torch
import torch.nn as nn


def double_conv(input_channels, output_channels) -> nn.Sequential:
    """
    Args:
        input_channels:
        output_channels:
    Returns:
    """
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True)
    )
    return conv


class UNet(nn.Module):
    """
    UNet class.
    """

    def __init__(self, input_channels, output_channels, n_classes) -> None:
        """
        Args:
            input_channels:
            output_channels:
            n_classes:
        """
        super(UNet, self).__init__()

        self.n_classes = n_classes

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(input_channels, output_channels)
        self.down_conv2 = double_conv(output_channels, 2 * output_channels)
        self.down_conv3 = double_conv(2 * output_channels, 4 * output_channels)
        self.down_conv4 = double_conv(4 * output_channels, 8 * output_channels)

        self.bootleneck = double_conv(8 * output_channels, 16 * output_channels)

        self.up_transpose4 = nn.ConvTranspose2d(16 * output_channels, 8 * output_channels, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(16 * output_channels, 8 * output_channels)
        self.up_transpose3 = nn.ConvTranspose2d(8 * output_channels, 4 * output_channels, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(8 * output_channels, 4 * output_channels)
        self.up_transpose2 = nn.ConvTranspose2d(4 * output_channels, 2 * output_channels, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(4 * output_channels, 2 * output_channels)
        self.up_transpose1 = nn.ConvTranspose2d(2 * output_channels, output_channels, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(2 * output_channels, output_channels)
        self.out_layer = nn.Conv2d(output_channels, out_channels=n_classes, kernel_size=1)

    def forward(self, image) -> torch.Tensor:
        """
        Args:
            image:
        Returns:
        """
        # Encoder part
        down1 = self.down_conv1(image)
        max_pool1 = self.max_pool_2x2(down1)
        down2 = self.down_conv2(max_pool1)
        max_pool2 = self.max_pool_2x2(down2)
        down3 = self.down_conv3(max_pool2)
        max_pool3 = self.max_pool_2x2(down3)
        down4 = self.down_conv4(max_pool3)
        max_pool4 = self.max_pool_2x2(down4)

        # Bootleneck
        bootleneck = self.bootleneck(max_pool4)

        # Decoder part
        up_transpose4 = self.up_transpose4(bootleneck)
        up_conv4 = self.up_conv4(torch.cat([up_transpose4, down4], dim=1))
        up_transpose3 = self.up_transpose3(up_conv4)
        up_conv3 = self.up_conv3(torch.cat([up_transpose3, down3], dim=1))
        up_transpose2 = self.up_transpose2(up_conv3)
        up_conv2 = self.up_conv2(torch.cat([up_transpose2, down2], dim=1))
        up_transpose1 = self.up_transpose1(up_conv2)
        up_conv1 = self.up_conv1(torch.cat([up_transpose1, down1], dim=1))
        out = self.out_layer(up_conv1)
        out = torch.sigmoid(out)

        return out
