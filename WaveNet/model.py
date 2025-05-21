import torch
import torch.nn as nn

class DilatedCausalConv1d(nn.Module):
    def __init__(self,
        res_channels,
        dilation
    ):
        super(DilatedCausalConv1d, self).__init__()
        self.pad = (2 - 1) * dilation # kernel size - 1 * dilation
        self.conv = nn.Conv1d(res_channels, res_channels, kernel_size=2, stride=1, padding=self.pad, dilation=dilation)
    
    def forward(self, x):
        return self.conv(x)[..., :-self.pad]
    
class CausalConv1d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        dilation
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation # kernel size - 1 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=self.pad, dilation=dilation)
    
    def forward(self, x):
        return self.conv(x)[..., :-self.pad]

class ResidualBlock(nn.Module):
    def __init__(self,
        res_channels,
        skip_channels,
        out_channels,
        dilation
    ):
        super(ResidualBlock, self).__init__()

        self.dilated_conv = DilatedCausalConv1d(res_channels, dilation)

        self.res_conv = nn.Conv1d(res_channels, res_channels, 1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        x: (batch_size, res_channels, time_steps)
        """
        x_dilated = self.dilated_conv(x) # (batch_size, res_channels, time_steps)
        x_gated = self.tanh(x_dilated) * self.sigmoid(x_dilated) # (batch_size, res_channels, time_steps)

        skip_output = self.skip_conv(x_gated) # (batch_size, skip_channels, time_steps)
        res_output = x + self.res_conv(x_gated) # (batch_size, res_channels, time_steps)

        return res_output, skip_output

class ResidualStack(nn.Module):
    def __init__(self,
        layer_num=10,
        block_size=4,
        res_channels=256,
        skip_channels=256,
    ):
        super(ResidualStack, self).__init__()

        self.dilation_rates = [2**i for i in range(layer_num)]
        self.dilation_rates = self.dilation_rates * block_size

        self.blocks = nn.ModuleList([ResidualBlock(res_channels, skip_channels, res_channels, self.dilation_rates[i]) for i in range(block_size * layer_num)])

    def forward(self, x):
        """
        x: (batch_size, res_channels, time_steps)
        """
        skip_connections = []
        for block in self.blocks:
            x, skip_connection = block(x) # (batch_size, res_channels, time_steps), (batch_size, skip_channels, time_steps)
            skip_connections.append(skip_connection)
        
        return torch.stack(skip_connections) # (block_size, batch_size, skip_channels, time_steps)

class WaveNet(nn.Module):
    def __init__(self,
        layers=10,
        blocks=4,
        classes=256,
        res_channels=32,
        skip_channels=512
    ):
        super(WaveNet, self).__init__()

        self.causal_conv = CausalConv1d(classes, res_channels, kernel_size=2, dilation=1)

        self.residual_stack = ResidualStack(layers, blocks, res_channels, skip_channels)

        self.out_conv_1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.out_conv_2 = nn.Conv1d(skip_channels, classes, 1)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1) # Removed: Softmax is part of CrossEntropyLoss

    def forward(self, x):
        """
        x: (batch_size, time_steps, classes) from one_hot_encode in train.py
        """
        # Transpose x to (batch_size, classes, time_steps) for Conv1d layers
        x = x.transpose(1, 2) 
        
        x = self.causal_conv(x) # (batch_size, res_channels, time_steps)

        skip_connections = self.residual_stack(x) # (block_size * layers, batch_size, skip_channels, time_steps)

        skip_connections = skip_connections.sum(dim=0) # (batch_size, skip_channels, time_steps)

        output = self.relu(self.out_conv_1(skip_connections)) # (batch_size, skip_channels, time_steps)
        output = self.out_conv_2(output) # (batch_size, classes, time_steps)

        # Transpose back to (batch_size, time_steps, classes) to match train.py's expected output shape for reshaping
        return output.transpose(1, 2)
        
if __name__ == "__main__":
    model = WaveNet(
        layers=10,
        blocks=4,
        classes=256,
        res_channels=256,
        skip_channels=256)
    
    x = torch.randn(1, 256, 1024) # 256 companding values, 10240 time steps
    output = model(x)
    print(output.shape) # (1, 256, 1024)