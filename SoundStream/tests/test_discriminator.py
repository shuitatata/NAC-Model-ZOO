from discriminator import ResidualUnit2D, STFT_Discriminator, WaveDiscriminator
import torch

def test_residual_unit_2d():
    x = torch.zeros(1, 32, 1024, 12) # (batch_size, in_channels, time, freq)
    unit = ResidualUnit2D(32, 2, 1, 2) # (N, m, st, sf)
    y = unit(x)
    assert y.shape == (1, 64, 1024, 6)

    x = torch.zeros(1, 32, 17352, 513) # (batch_size, in_channels, time, freq)
    y = unit(x)
    assert y.shape == (1, 64, 17352, 257)

    x = torch.zeros(1, 32, 1024, 12)
    unit = ResidualUnit2D(32, 2, 2, 2) # (N, m, st, sf)
    y = unit(x)
    assert y.shape == (1, 64, 512, 6)

def test_stft_discriminator():
    x = torch.zeros(1, 2, 1024, 1024) # (batch_size, in_channels, time, freq)
    discriminator = STFT_Discriminator(32, 1024) # (C, F)
    y = discriminator(x)
    assert len(y) == 8
    assert y[-1].shape == (1, 1, 1024 // 8, 1) # (batch_size, out_channels, time, freq)

def test_stft_discriminator_lengths():
    discriminator = STFT_Discriminator(32, 1024) # (C, F)
    x = torch.zeros(1, 2, 1024, 1024) # (batch_size, in_channels, time, freq)
    lengths = discriminator.cal_lengths(1024)
    y = discriminator(x)
    assert len(y) == 8
    for i in range(7):
        assert y[i].shape[2] == lengths[i]

def test_wave_discriminator():
    x = torch.zeros(1, 1, 1024) # (batch_size, in_channels, time)
    discriminator = WaveDiscriminator(3) # (num_D)
    y = discriminator(x)
    assert len(y) == 3
    assert len(y[0]) == 7
    assert len(y[1]) == 7
    assert len(y[2]) == 7
    assert y[0][-1].shape == (1, 1, 1024 // 4**4) # (batch_size, out_channels, time)
    assert y[1][-1].shape == (1, 1, 1024 // 4**4 // 2) # (batch_size, out_channels, time)
    assert y[2][-1].shape == (1, 1, 1024 // 4**4 // 4) # (batch_size, out_channels, time)

def test_wave_discriminator_lengths():
    discriminator = WaveDiscriminator(3) # (num_D)
    x = torch.zeros(1, 1, 1024) # (batch_size, in_channels, time)
    lengths = discriminator.cal_lengths(1024)
    y = discriminator(x)
    assert len(y) == 3
    for i in range(3):
        assert len(y[i]) == 7
        for j in range(7):
            assert y[i][j].shape[2] == lengths[i][j]