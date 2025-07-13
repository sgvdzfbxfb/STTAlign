import torch
import torch.nn as nn

def PositionalEncoding(max_len, d_model, device):
    encoding = torch.zeros(max_len, d_model, device=device)
    encoding.requires_grad = False
    pos = torch.arange(0, max_len, device=device)
    pos = pos.float().unsqueeze(dim=1)
    _2i = torch.arange(0, d_model, step=2, device=device).float()
    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    return  encoding