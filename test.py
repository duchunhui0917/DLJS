import torch
from torch.distributions import Categorical
import torch.nn.functional as F

logits = torch.tensor([[1., 9.],
                       [2., 8.],
                       [3., 7.]])
print(F.softmax(logits, dim=1))
