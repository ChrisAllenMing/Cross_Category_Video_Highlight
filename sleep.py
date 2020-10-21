import time
import torch

while True:
    x = torch.rand((100, 2048)).cuda()
    y = torch.rand((100, 2048)).cuda()
    z = torch.mm(x, y.t()).mean()
    print('Results: ', z.item())
    time.sleep(10)