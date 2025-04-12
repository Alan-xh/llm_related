<<<<<<< HEAD
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16)

x1, x2 = x.chunk(2, dim=-1)

print(x1)
print(x2)

print(x1.shape)
print(x2.shape)
x3 = torch.cat((x1, x2), dim=-1)

print(x3)

print(x3.unsqueeze(0).shape)
=======
from transformers.models.gpt2 import GPT2Model

import torch


x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(x.shape)
x1, x2 = x.chunk(2, dim=-1)
print(x1, x2)
print(torch.cat((-x2, x1), dim=-1))
>>>>>>> refs/remotes/origin/main
