from SoundStream.encoder import Encoder
from SoundStream.decoder import Decoder
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

class Generator(nn.Module):
    def __init__(self, C_enc, C_dec, D, n_q, codebook_size):
        super(Generator, self).__init__()
        self.strides = [2,4,5,8]
        self.encoder = Encoder(C_enc, D, self.strides)
        self.decoder = Decoder(C_dec, D, self.strides[::-1])
        self.RVQ = ResidualVQ(
            dim = D,
            codebook_size=codebook_size,
            num_quantizers=n_q,
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        print(f"encoder output: {embeddings.shape}")
        
        # 转置: (batch, channels, length) -> (batch, length, channels)
        embeddings = embeddings.transpose(1, 2)
        print(f"before RVQ: {embeddings.shape}")
        
        embeddings, _, _ = self.RVQ(embeddings)
        print(f"after RVQ: {embeddings.shape}")
        
        # 转置回来: (batch, length, channels) -> (batch, channels, length)
        embeddings = embeddings.transpose(1, 2)
        print(f"before decoder: {embeddings.shape}")
        
        out = self.decoder(embeddings)
        return out

