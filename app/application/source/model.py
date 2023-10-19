import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, input_size=24):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out
