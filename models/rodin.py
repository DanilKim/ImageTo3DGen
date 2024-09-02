import torch
import torch.nn as nn


class Rodin(nn.Module):
    def __init__(self, opt, device):
        super().__init__()