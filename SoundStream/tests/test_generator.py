from SoundStream.generator import Generator
import torch

def test_generator():
    generator = Generator(32, 32, 1024, 4, 1024)
    x = torch.zeros(1, 1, 6400)
    y = generator(x)
    assert y.shape == (1, 1, 6400)