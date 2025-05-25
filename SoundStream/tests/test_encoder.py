from SoundStream.encoder import CausalConv1d, EncoderResidualUnit, EncoderBlock, Encoder
import torch

def test_causal_conv1d_wo_dilation_shape():
    x = torch.randn(1, 10, 20) # (batch_size, in_channels, seq_len)
    conv = CausalConv1d(10, 5, 1, dilation=1)
    y = conv(x)

    # check shape
    assert y.shape == (1, 5, 20)

def test_causal_conv1d_wo_dilation_causal():
    # check causal
    x = torch.zeros(1, 1, 20) # (batch_size, in_channels, seq_len)
    conv = CausalConv1d(1, 1, 1, dilation=1, bias = False)
    x[:, :, 10:] = 1
    y = conv(x)
    assert y.shape == (1, 1, 20)
    print(y)
    assert torch.all(y[:, :, :10] == False)

def test_causal_conv1d_with_dilation_shape():
    x = torch.randn(1, 10, 20) # (batch_size, in_channels, seq_len)
    conv = CausalConv1d(10, 5, 3, dilation=2)
    y = conv(x)
    assert y.shape == (1, 5, 20)

def test_causal_conv1d_with_dilation_causal():
    x = torch.zeros(1, 1, 20) # (batch_size, in_channels, seq_len)
    conv = CausalConv1d(1, 1, 1, dilation=2, bias = False)
    x[:, :, 10:] = 1
    y = conv(x)
    assert y.shape == (1, 1, 20)
    assert torch.all(y[:, :, :10] == 0)

def test_encoder_residual_unit():
    x = torch.randn(1, 10, 20) # (batch_size, in_channels, seq_len)
    conv = EncoderResidualUnit(10) # in_channels=10, out_channels=5
    y = conv(x) # (batch_size, out_channels, seq_len)
    assert y.shape == (1, 10, 20)

def test_encoder_block():
    x = torch.randn(1, 10, 20) # (batch_size, in_channels, seq_len)
    conv = EncoderBlock(10, 20) # in_channels=10, out_channels=20
    y = conv(x) # (batch_size, out_channels, seq_len)
    assert y.shape == (1, 20, 20)

def test_encoder():
    x = torch.randn(1, 10, 640) # (batch_size, in_channels, seq_len)
    conv = Encoder(10, 20, [2,4,5,8]) # in_channels=10, out_channels=20
    # M = 2*4*5*8 = 320
    y = conv(x) # (batch_size, out_channels, 640/M)
    assert y.shape == (1, 20, 2)