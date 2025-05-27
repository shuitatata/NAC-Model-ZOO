from generator import Generator
import torch

# def test_generator():
#     generator = Generator(32, 32, 1024, 4, 1024)
#     x = torch.zeros(1, 1, 572945)
#     y = generator(x)
#     assert y.shape[0] == 1
#     assert y.shape[1] == 1
#     assert y.shape[2] // 320 == x.shape[2] // 320
