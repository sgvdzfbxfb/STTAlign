import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, segment_csr
from net.vitnet import mae_vit_base_patch16
from pytorch3d.transforms import *

class teeth_reship():
    def __init__(self):
        super(teeth_reship, self).__init__()
        self.edge_index = torch.tensor(
        [[1, 2], [1, 16],
        [2, 1],   [2, 3],   [2, 15],
        [3, 2],   [3, 4],   [3, 14],
        [4, 3],   [4, 5],   [4, 13],
        [5, 4],   [5, 6],   [5, 12],
        [6, 5],   [6, 7],   [6, 11],
        [7, 6],   [7, 8],   [7, 10],
        [8, 7],   [8, 9],   [8, 9],
        [9, 8],   [9, 10],  [9, 8],
        [10, 9],  [10, 11], [10, 7],
        [11, 10], [11, 12], [11, 6],
        [12, 11], [12, 13], [12, 5],
        [13, 12], [13, 14], [13, 4],
        [14, 13], [14, 15], [14, 3],
        [15, 14], [15, 16], [15, 2],
        [16, 15], [16, 1],
        [17, 18],  [17, 32],
        [18, 17],  [18, 19], [18, 31],
        [19, 18],  [19, 20], [19, 30],
        [20, 19],  [20, 21], [20, 29],
        [21, 20],  [21, 22], [21, 28],
        [22, 21],  [22, 23], [22, 27],
        [23, 22],  [23, 24], [23, 26],
        [24, 23],  [24, 25], [24, 25],
        [25, 24],  [25, 26], [25, 24],
        [26, 25],  [26, 27], [26, 23],
        [27, 26],  [27, 28], [27, 22],
        [28, 27],  [28, 29], [28, 21],
        [29, 28],  [29, 30], [29, 20],
        [30, 29],  [30, 31], [30, 19],
        [31, 30],  [31, 32], [31, 18],
        [32, 31],  [32, 17]])

class teeth_arangement_model(nn.Module):
    def __init__(self, in_channel=3):
        super(teeth_arangement_model, self).__init__()
        self.local_fea = mae_vit_base_patch16()

    def forward(self, teeth_points, centerp):
        B,T,N,C = teeth_points.shape
        teeth_points = (teeth_points - centerp).permute(0, 1, 3, 2)
        cenv = centerp.permute(0, 1, 3, 2).expand(-1,-1,-1, N)
        tmaxv = torch.max(torch.abs(teeth_points.reshape(B, T, N*C)), dim=-1)[0].view(B, T, 1, 1)
        teeth_points = torch.cat([teeth_points, cenv], dim=-2)
        local_dofs = []
        local_tranvs = []
        for bid in range(teeth_points.shape[0]):
            local_dofs_, trans_feat = self.local_fea(teeth_points[bid, :, :, :])
            local_dofs.append(local_dofs_)
            local_tranvs.append(trans_feat)
        local_dofs = torch.stack(local_dofs, dim=0)
        local_tranvs = torch.stack(local_tranvs, dim=0)
        pdofs = local_dofs
        output = local_tranvs
        return pdofs, output