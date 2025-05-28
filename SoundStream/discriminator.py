import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import torch


class ResidualUnit2D(nn.Module):
    def __init__(self, N, m, st, sf):
        super(ResidualUnit2D, self).__init__()
        self.st = st
        self.sf = sf
        self.layers = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=N,
                      out_channels=N, padding='same'),
            nn.ELU(),
            nn.Conv2d(kernel_size=(st+2, sf+2), in_channels=N,
                      out_channels=N * m, stride=(st, sf)),
        )

        self.skip_conv = nn.Conv2d(kernel_size=(
            1, 1), in_channels=N, out_channels=N * m, stride=(st, sf))

    def forward(self, x):
        return self.layers(F.pad(x, [self.sf+1, 0, self.st+1, 0])) + self.skip_conv(x)


class STFT_Discriminator(nn.Module):
    def __init__(self, C, F):
        super(STFT_Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=C, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2D(C, 2, 1, 2),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2D(2*C, 2, 2, 2),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2D(4*C, 1, 1, 2),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2D(4*C, 2, 2, 2),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2D(8*C, 1, 1, 2),
                nn.ELU(),
            ),
            nn.Sequential(
                ResidualUnit2D(8*C, 2, 2, 2),
                nn.ELU(),
            ),
            nn.Conv2d(kernel_size=(1, F//64), in_channels=16 *
                      C, out_channels=1, stride=1),
        ])

    def cal_lengths(self, input_length):
        return [
            input_length-6,
            input_length-6,
            torch.div(input_length-5, 2, rounding_mode='floor'),
            torch.div(input_length-5, 2, rounding_mode='floor'),
            torch.div(input_length-3, 4, rounding_mode='floor'),
            torch.div(input_length-3, 4, rounding_mode='floor'),
            torch.div(input_length+1, 8, rounding_mode='floor'),
            torch.div(input_length+1, 8, rounding_mode='floor'),
        ]

    def forward(self, x):
        results = []
        for layer in self.layers:
            x = layer(x)
            results.append(x)
        return results


# Wave-Discriminator，ref:https://github.com/descriptinc/melgan-neurips
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])
    
    def cal_lengths(self, input_length):
        return [
            input_length,
            torch.div(input_length+3, 4, rounding_mode='floor'),
            torch.div(input_length+15, 16, rounding_mode='floor'),
            torch.div(input_length+63, 64, rounding_mode='floor'),
            torch.div(input_length+255, 256, rounding_mode='floor'),
            torch.div(input_length+255, 256, rounding_mode='floor'),
            torch.div(input_length+255, 256, rounding_mode='floor'),
        ]

    def forward(self, x):
        results = []
        for layer in self.layers:
            x = layer(x)
            results.append(x)
        return results

class WaveDiscriminator(nn.Module):
    def __init__(self, num_D=3):
        super(WaveDiscriminator, self).__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([WaveDiscriminatorBlock() for _ in range(num_D)])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        results = []
        for discriminator in self.discriminators:
            results.append(discriminator(x))
            x = self.downsample(x)
        return results

    def cal_lengths(self, input_length):
        lengths = []
        for i in range(self.num_D):
            length = torch.div(input_length, 2**i, rounding_mode='floor')
            lengths.append(self.discriminators[i].cal_lengths(length))
        return lengths