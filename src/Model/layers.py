import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, n_blocks=1, activation: nn.Module = nn.ReLU()):
        super(ConvBlock, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3),
                nn.BatchNorm2d(out_dim),
                activation
            ) for _ in range(n_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):
    def __init__(self, dims, dropout=0., activation=nn.GELU()):
        super(MLP, self).__init__()
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                activation,
                nn.Dropout(dropout)
            ) for i in range(len(dims) - 2)],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        y = self.blocks(x)
        return y
