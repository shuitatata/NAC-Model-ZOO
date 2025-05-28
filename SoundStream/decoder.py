import torch
from encoder import ResidualUnit, CausalConv1d
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class TransposedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(TransposedCausalConv1d, self).__init__()
        
        self.causal_padding = (kernel_size - 1) * dilation
        self.output_padding = stride - 1
        self.causal_output_padding = (kernel_size - 1) * dilation - self.output_padding
        
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, dilation=dilation, bias=bias
        )

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len)
        """
        output = self.conv(x)
        if self.causal_output_padding > 0:
            output = output[..., :-self.causal_output_padding]
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DecoderBlock, self).__init__()

        self.layers = nn.Sequential(
            TransposedCausalConv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2 * stride, stride=stride),
            nn.ELU(),
            ResidualUnit(out_channels, dilation=1),
            nn.ELU(),
            ResidualUnit(out_channels, dilation=3),
            nn.ELU(),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x):
        logger.debug(f"DecoderBlock: {x.shape}")
        x = self.layers(x)
        logger.debug(f"DecoderBlock-output shape: {x.shape}")
        return x

class Decoder(nn.Module):
    """
    in_channels: must can be divided by 16
    x: (batch_size, in_channels, seq_len)
    return: (batch_size, 1, seq_len)
    """
    def __init__(self, hidden_channels, in_channels, strides=[8,5,4,2]):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels, hidden_channels, kernel_size=7),
            nn.ELU(),
            DecoderBlock(hidden_channels, hidden_channels // 2, stride=strides[0]),
            nn.ELU(),
            DecoderBlock(hidden_channels // 2, hidden_channels // 4, stride=strides[1]),
            nn.ELU(),
            DecoderBlock(hidden_channels // 4, hidden_channels // 8, stride=strides[2]),
            nn.ELU(),
            DecoderBlock(hidden_channels // 8, hidden_channels // 16, stride=strides[3]),
            nn.ELU(),
            CausalConv1d(hidden_channels // 16, 1, kernel_size=7)
        )
        
    def forward(self, x):
        return self.layers(x)
        