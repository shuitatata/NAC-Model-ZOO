import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit2D(nn.Module):
    def __init__(self, N, m, st, sf):
        super(ResidualUnit2D, self).__init__()
        self.st = st
        self.sf = sf
        self.layers = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=N, out_channels=N, padding='same'),
            nn.ELU(),
            nn.Conv2d(kernel_size=(st+2, sf+2), in_channels=N, out_channels=N * m, stride=(st, sf), padding=(1, 1)),
            nn.ELU(),
        )

        self.skip_conv = nn.Conv2d(kernel_size=(1, 1), in_channels=N, out_channels=N * m, stride=(st, sf))

    def forward(self, x):
        
        return self.layers(x) + self.skip_conv(x)

class STFT_Discriminator(nn.Module):
    def __init__(self, C, F):
        super(STFT_Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ResidualUnit2D(C, 2, 1, 2),
            nn.ELU(),
            ResidualUnit2D(2*C, 2, 2, 2), # time: 1024 -> 512
            nn.ELU(),
            ResidualUnit2D(4*C, 1, 1, 2), # time: 512 -> 512
            nn.ELU(),
            ResidualUnit2D(4*C, 2, 2, 2), # time: 512 -> 256
            nn.ELU(),
            ResidualUnit2D(8*C, 1, 1, 2), # time: 256 -> 256
            nn.ELU(),
            ResidualUnit2D(8*C, 2, 2, 2), # time: 256 -> 128
            nn.ELU(),
            nn.Conv2d(kernel_size=(1, F//64), in_channels=16*C, out_channels=1, stride=1),
            nn.ELU(),
        )
    
    def forward(self, x):
        return self.layers(x)
    
    