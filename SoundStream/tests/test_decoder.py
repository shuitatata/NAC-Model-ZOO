import os
print(os.sys.path)
import torch
from decoder import TransposedCausalConv1d, DecoderBlock, Decoder

def test_transposed_causal_conv1d_shape():
    x = torch.zeros(1, 5, 20) # (batch_size, in_channels, seq_len)
    conv = TransposedCausalConv1d(in_channels=5, out_channels=20, kernel_size=1, dilation=1, bias=False)
    y = conv(x)
    assert y.shape == (1, 20, 20)

    x = torch.zeros(1, 1, 20) # (batch_size, in_channels=1, seq_len)
    conv = TransposedCausalConv1d(in_channels=1, out_channels=5, kernel_size=3, dilation=1, stride=3, bias=False)
    y = conv(x)
    assert y.shape == (1, 5, 60)

    conv = TransposedCausalConv1d(in_channels=1, out_channels=5, kernel_size=3, dilation=3, stride=3, bias=False)
    y = conv(x)
    assert y.shape == (1, 5, 60)

    x = torch.zeros(1, 1, 37) # (batch_size, in_channels=1, seq_len)
    conv = TransposedCausalConv1d(in_channels=1, out_channels=5, kernel_size=3, dilation=3, stride=3, bias=False)
    y = conv(x)
    assert y.shape == (1, 5, 111)

def test_transposed_causal_conv1d_causal():
    x = torch.zeros(1, 5, 20) # (batch_size, in_channels, seq_len)
    x[:, :, 10:] = 1
    conv = TransposedCausalConv1d(in_channels=5, out_channels=20, kernel_size=1, dilation=1, bias=False)
    y = conv(x)
    assert y.shape == (1, 20, 20)
    assert torch.all(y[:, :, :10] == 0)
    assert not torch.any(y[:, :, 10:] == 0)

def test_decoder_block():
    x = torch.zeros(1, 5, 20) # (batch_size, in_channels, seq_len)
    block = DecoderBlock(5, 20, stride=1)
    y = block(x)
    assert y.shape == (1, 20, 20)

def test_decoder():
    x = torch.zeros(1, 5, 2) # (batch_size, in_channels, seq_len)
    decoder = Decoder(32, 5, strides=[8, 5, 4, 2])
    # M = 2 * 4 * 5 * 8 = 320
    y = decoder(x)
    assert y.shape == (1, 1, 640)

