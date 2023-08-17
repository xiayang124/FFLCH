import torch
import numpy

a = [[False, True, False, True, False, True, True],
     [True, False, True, False, True, False, True]]
b = [[[0, 1, 2, 3, 4, 5, 6],
     [7, 8, 9, 10, 11, 12, 13]]]

a = torch.tensor(a)
b = torch.tensor(b)

print(a.shape)
print(b.shape)

a = b[:, a]
print(0)
