import torch
import torch.nn as nn


class WrappedBlock(nn.Module):
    def __init__(self, block):
        super(WrappedBlock, self).__init__()
        self.block = block
        children = list(self.block.mlp.children())
        self.mlp_p1 = nn.Sequential(*children[:-2])
        self.mlp_p2 = nn.Sequential(*children[-2:])
        self.ogx = []
        self.sample_ind = 0

    def clear(self):
        self.ogx = []

    def cat(self):
        self.ogx = torch.cat(self.ogx, 0)

    def forward_p1(self, x: torch.Tensor) -> torch.Tensor:
        self.ogx.append(x.clone())
        # print(x.shape)
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = self.mlp_p1(self.block.norm2(x))
        return x

    def forward_p2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ogx[self.sample_ind] + self.block.drop_path2(self.block.ls2(self.mlp_p2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_p1(x)
        x = self.forward_p2(x)
        return x