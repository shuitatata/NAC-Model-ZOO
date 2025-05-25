import torch
from vector_quantize_pytorch import ResidualVQ
import torch.nn.functional as F
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)  # (batch_size, out_channels, seq_len)


class EncoderResidualUnit(nn.Module):
    """
        The first Conv1d's kernel size is 7, channels is out_channels, stride is 1
        The second Conv1d's kernel size is 1, channels is out_channels, stride is 1
    """

    def __init__(self, channels, dilation=1):
        super(EncoderResidualUnit, self).__init__()

        self.layers = nn.Sequential(
            CausalConv1d(channels, channels,
                         kernel_size=7, dilation=dilation),
            nn.ELU(),
            CausalConv1d(channels, channels,
                         kernel_size=1, dilation=1),
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        return x + self.layers(x)  # (batch_size, out_channels, seq_len)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncoderBlock, self).__init__()
        self.layers = nn.Sequential(
            EncoderResidualUnit(in_channels, dilation=1),
            nn.ELU(),
            EncoderResidualUnit(in_channels, dilation=3),
            nn.ELU(),
            EncoderResidualUnit(in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels, out_channels,
                         kernel_size=2*stride, stride=stride),
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        return self.layers(x)  # (batch_size, out_channels, seq_len)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, strides=[2, 4, 5, 8]):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels, in_channels, kernel_size=7),
            nn.ELU(),
            EncoderBlock(in_channels, in_channels * 2, stride=strides[0]),
            nn.ELU(),
            EncoderBlock(in_channels * 2, in_channels * 4, stride=strides[1]),
            nn.ELU(),
            EncoderBlock(in_channels * 4, in_channels * 8, stride=strides[2]),
            nn.ELU(),
            EncoderBlock(in_channels * 8, in_channels * 16, stride=strides[3]),
            nn.ELU(),
            CausalConv1d(in_channels * 16, out_channels, kernel_size=3)
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        return self.layers(x) # (batch_size, out_channels, seq_len)
