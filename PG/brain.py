import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_input, 3)
        self.l2 = nn.Linear(3, 9)
        self.l3 = nn.Linear(9, 18)
        self.l4 = nn.Linear(18, 9)
        self.l5 = nn.Linear(9, n_output)

    def forward(self, state):
        x1 = torch.tanh(self.l1(state))
        x2 = torch.tanh(self.l2(x1))
        x3 = torch.tanh(self.l3(x2))
        x4 = torch.tanh(self.l4(x3))
        x5 = self.l5(x4)
        return torch.unsqueeze(torch.squeeze(x5, dim=-1), dim=0)
