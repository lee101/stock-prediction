from torch import nn
import torch

from SCINet.models import SCINet
from SCINet.models.SCINet import SCINet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.relu = nn.ReLU() adding relu seems to hurt accuracy
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to('cuda')
        out, (hn) = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def get_model(input_len, input_dim=9, hidden_dim=1, num_stacks=1, output_dim=1):
    model = SCINet(output_len=output_dim, input_len=input_len, input_dim=input_dim, hid_size=hidden_dim, num_stacks=1,
                   num_levels=3, concat_len=0, groups=1, kernel=3, dropout=.5,
                   single_step_output_One=0, positionalE=True, modified=True,
                   RIN=False, # todo try rin
                   )
    model.to(DEVICE)
    return model
