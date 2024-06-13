import msadapter.pytorch.nn as nn
# import torch.nn as nn

class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.hidden = nn.ModuleList()
        self.hidden.append(nn.Linear(3, 64))  # 输入为3个特征
        for _ in range(8):
            self.hidden.append(nn.Linear(64, 64))
        self.output = nn.Linear(64, 3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.hidden:
            x = self.tanh(layer(x))
        return self.output(x)
