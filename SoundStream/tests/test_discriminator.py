from SoundStream.discriminator import ResidualUnit2D, STFT_Discriminator
import torch

def test_residual_unit_2d():
    x = torch.zeros(1, 32, 1024, 12) # (batch_size, in_channels, time, freq)
    unit = ResidualUnit2D(32, 2, 1, 2) # (N, m, st, sf)
    y = unit(x)
    assert y.shape == (1, 64, 1024, 6)

    unit = ResidualUnit2D(32, 2, 2, 2) # (N, m, st, sf)
    y = unit(x)
    assert y.shape == (1, 64, 512, 6)

def test_stft_discriminator():
    x = torch.zeros(1, 32, 1024, 1024) # (batch_size, in_channels, time, freq)
    discriminator = STFT_Discriminator(32, 1024) # (C, F)
    y = discriminator(x)
    assert y.shape == (1, 1, 1024 // 8, 1) # (batch_size, out_channels, time, freq)