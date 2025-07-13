import numpy as np
import torch
import torch.nn.functional as F
from config import config as cfg

def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from pytorch3d.transforms import *

class Tooth_Assembler(nn.Module):
    def __init__(self):
        super(Tooth_Assembler, self).__init__()


    def forward(self,pred: torch.Tensor, cenp: torch.Tensor, dofs: torch.Tensor, ptrans: torch.Tensor, device: torch.device) -> torch.Tensor:

        assembled = torch.zeros(size=pred.shape, device=device)
        pred_matrices = torch.cat([quaternion_to_matrix(dofs[idx]).unsqueeze(0) for idx in range(dofs.shape[0])], dim=0)

        pred_matrices_numpy = pred_matrices.detach().cpu().numpy()
        for idx in range(pred.shape[0]):

            centerp = cenp[idx, :, :, :]
            points = pred[idx, :, :, :] - centerp

            transv = ptrans[idx, :, :].unsqueeze(1)
            points = torch.bmm(points, pred_matrices[idx, :, :, :])

            assembled[idx, :, :, :] = points + transv
            assembled[idx, :, :, :] = assembled[idx, :, :, :]  + centerp

        return assembled