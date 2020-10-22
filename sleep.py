import time
import torch

while True:
    x = torch.rand((4000, 2048)).cuda()
    y = torch.rand((4000, 2048)).cuda()
    z = torch.mm(x, y.t()).mean()
    print('Results: ', z.item())