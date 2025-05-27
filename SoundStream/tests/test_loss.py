from loss_func import G_adversarial_loss, D_adversarial_loss, G_feature_loss, G_rec_loss, G_criterion
import torch
import config

def test_G_adversarial_loss():
    x_hat = torch.zeros(10, 1, 1024) # (batch_size, in_channels, time)
    stft_output = torch.ones(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    wave_output = [torch.ones(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    stft_output_lengths = torch.tensor([1024] * 10) # [batch_size]
    wave_output_lengths = torch.tensor([[1024] * 10] * 3) # [n_discriminator, batch_size]
    loss = G_adversarial_loss(stft_output, stft_output_lengths, wave_output, wave_output_lengths)

    assert loss.shape == ()
    assert loss.item() == 0

    stft_output = torch.zeros(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    wave_output = [torch.zeros(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    loss = G_adversarial_loss(stft_output, stft_output_lengths, wave_output, wave_output_lengths)
    assert loss.shape == ()
    assert loss.item() ==1

def test_D_adversarial_loss():
    stft_output_x = torch.ones(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    stft_output_x_hat = -torch.ones(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    stft_output_lengths = torch.tensor([1024] * 10) # [batch_size]

    wave_output_x = [torch.ones(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    wave_output_x_hat = [-torch.ones(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    wave_output_lengths = torch.tensor([[1024] * 10] * 3) # [n_discriminator, batch_size]

    loss = D_adversarial_loss(stft_output_x, stft_output_x_hat, stft_output_lengths, wave_output_x, wave_output_x_hat, wave_output_lengths)

    assert loss.shape == ()
    assert loss.item() == 0


    stft_output_x = -torch.ones(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    stft_output_x_hat = torch.ones(10, 1, 1024, 1) # (batch_size, out_channels, time, freq)
    wave_output_x = [-torch.ones(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    wave_output_x_hat = [torch.ones(10, 1, 1024) for _ in range(3)] # (batch_size, out_channels, time, freq)
    loss = D_adversarial_loss(stft_output_x, stft_output_x_hat, stft_output_lengths, wave_output_x, wave_output_x_hat, wave_output_lengths)
    assert loss.shape == ()
    assert loss.item() != 0

def test_G_feature_loss():
    # STFT features: [batch_size, channels, time, freq]
    stft_outputs_x = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32)]
    stft_outputs_x_hat = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32)]
    stft_output_lengths = [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50])]

    # Wave features: list of lists [batch_size, channels, time, freq]
    wave_outputs_x = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128)],  # disc 0
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64)]    # disc 1
    ]
    wave_outputs_x_hat = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128)],  # disc 0
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64)]    # disc 1
    ]
    wave_output_lengths = [
        [torch.tensor([256, 240, 220, 200]), torch.tensor([128, 120, 110, 100])],  # disc 0
        [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50])]       # disc 1
    ]

    loss = G_feature_loss(stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, 
                         wave_outputs_x, wave_outputs_x_hat, wave_output_lengths)

    assert loss.shape == ()
    assert loss.item() == 0

    # Test with different features
    stft_outputs_x_hat = [torch.zeros(4, 32, 128, 64), torch.zeros(4, 64, 64, 32)]
    wave_outputs_x_hat = [
        [torch.zeros(4, 16, 256), torch.zeros(4, 32, 128)],
        [torch.zeros(4, 16, 128), torch.zeros(4, 32, 64)]
    ]
    
    loss = G_feature_loss(stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, 
                         wave_outputs_x, wave_outputs_x_hat, wave_output_lengths)
    
    assert loss.shape == ()
    assert loss.item() > 0

def test_G_rec_loss():
    x = torch.zeros(10, 1, 10240) # (batch_size, 1, time)
    x_hat = torch.ones(10, 1, 10240) # (batch_size, 1, time)
    loss = G_rec_loss(x, x_hat)
    assert loss.shape == ()
    assert loss.item() > 0

    x = torch.ones(10, 1, 10240) # (batch_size, 1, time)
    loss = G_rec_loss(x, x)
    assert loss.shape == ()
    assert loss.item() == 0

def test_G_criterion():
    x = torch.ones(4, 1, 10240) # (batch_size, 1, time)
    x_hat = torch.ones(4, 1, 10240) # (batch_size, 1, time)
    stft_outputs_x = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32), torch.ones(4, 1, 72, 1)]
    stft_outputs_x_hat = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32), torch.ones(4, 1, 72, 1)]
    stft_output_lengths = [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50]), torch.tensor([72, 68, 64, 60])]
    
    wave_outputs_x = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128), torch.ones(4, 1, 144)],
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64), torch.ones(4, 1, 72)]
    ]
    wave_outputs_x_hat = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128), torch.ones(4, 1, 144)],
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64), torch.ones(4, 1, 72)]
    ]
    wave_output_lengths = [
        [torch.tensor([256, 240, 220, 200]), torch.tensor([128, 120, 110, 100]), torch.tensor([144, 140, 136, 132])],
        [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50]), torch.tensor([72, 68, 64, 60])]
    ]

    loss = G_criterion(config,x, x_hat, stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, wave_outputs_x, wave_outputs_x_hat, wave_output_lengths)
    assert loss.shape == ()
    assert loss.item() == 0

def test_D_criterion():
    x = torch.ones(4, 1, 10240) # (batch_size, 1, time)
    x_hat = torch.ones(4, 1, 10240) # (batch_size, 1, time)
    stft_outputs_x = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32), torch.ones(4, 1, 72, 1)]
    stft_outputs_x_hat = [torch.ones(4, 32, 128, 64), torch.ones(4, 64, 64, 32), torch.ones(4, 1, 72, 1)]
    stft_output_lengths = [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50]), torch.tensor([72, 68, 64, 60])]
    
    wave_outputs_x = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128), torch.ones(4, 1, 144)],
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64), torch.ones(4, 1, 72)]
    ]
    wave_outputs_x_hat = [
        [torch.ones(4, 16, 256), torch.ones(4, 32, 128), torch.ones(4, 1, 144)],
        [torch.ones(4, 16, 128), torch.ones(4, 32, 64), torch.ones(4, 1, 72)]
    ]
    wave_output_lengths = [
        [torch.tensor([256, 240, 220, 200]), torch.tensor([128, 120, 110, 100]), torch.tensor([144, 140, 136, 132])],
        [torch.tensor([128, 120, 110, 100]), torch.tensor([64, 60, 55, 50]), torch.tensor([72, 68, 64, 60])]
    ]

    loss = G_criterion(config,x, x_hat, stft_outputs_x, stft_outputs_x_hat, stft_output_lengths, wave_outputs_x, wave_outputs_x_hat, wave_output_lengths)