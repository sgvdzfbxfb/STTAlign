import os
import copy
import numpy as np
import torch
import torch.nn as nn
import vtkmodules.all as vtk
from data.utils import get_files, walkFile, rotate_maxtrix
from config import config as cfg
from pytorch3d.transforms import *
from pytorch3d.transforms.se3 import se3_log_map
from pytorch3d.transforms import quaternion_to_axis_angle
import math
from typing import Dict, List, Optional, Tuple, Callable

class TrainDataAugmentation(nn.Module):
    def __init__(self):
        super(TrainDataAugmentation, self).__init__()

    @torch.no_grad()
    def forward(self, X: torch.Tensor):
        teeth_num = X.shape[0]
        trans = Transform3d().compose(Translate(-X["C"]),
                                      Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (cfg.teeth_nums, 3)), "XYZ")),
                                      Translate(torch.clamp(torch.randn(cfg.teeth_nums, 3), -3.14, 3.14)),
                                      Translate(X["C"]))
        X = trans.transform_points(X["X_v"])
        X = X.clone().reshape(shape=X["X"].shape)
        X_matrices = trans.inverse().get_matrix()
        final_trans_mat = X_matrices
        X["6dof"] = se3_log_map(final_trans_mat)
        return X

def data_load_no_centering(file_path):
    file_data = np.load(file_path, allow_pickle=True).item()
    teeth_nums = []
    teeth_points = []
    for key in file_data:
        teeth_nums.append(int(key))
        teeth_points.append(file_data[key])
    teeth_nums = np.array(teeth_nums)
    order_index = np.argsort(teeth_nums)
    teeth_nums = teeth_nums[order_index]
    teeth_points = np.array(teeth_points)
    teeth_points = teeth_points[order_index]
    return teeth_points, teeth_nums

def data_load_collision(file_path):
    file_data = np.load(file_path, allow_pickle=True).item()
    teeth_nums = []
    teeth_col_points = []
    for key in file_data:
        teeth_nums.append(int(key))
        teeth_col_points.append(file_data[key])
    teeth_nums = np.array(teeth_nums)
    order_index = np.argsort(teeth_nums)
    teeth_nums = teeth_nums[order_index]
    teeth_col_points = np.array(teeth_col_points)
    teeth_col_points = teeth_col_points[order_index]
    return teeth_col_points

def teeth_whole_rotate(teeth_points, rt):
    teeth_points = teeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    teeth_points = (rt.dot(teeth_points.T)).T
    teeth_points = teeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
    return teeth_points

class TrainData():
    def __init__(self, file_root):
        self.data_dir = file_root
        self.train_list = None
        self.prepare(self.data_dir)
    def prepare(self, file_path):
        file_list = []
        get_files(file_path, file_list, "down_end.npy")
        file_list_real = []
        for fl in file_list:
            corrs_up_dir = fl.replace("_down", "_up")
            if os.path.exists(corrs_up_dir):
                file_list_real.append(fl)
        self.train_list = file_list_real
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, item):
        file_path = self.train_list[item]
        return file_path

def train_data_load(file_path_):
    tRteeth_points, tGteeth_points, tteeth_center, tgdofs, ttrans_mats, tweights, rweights, rcpoints, mask_ = [],[],[],[],[],[],[],[],[]
    for ffi in range(len(file_path_)):
        file_path = file_path_[ffi]
        Gteeth_points, teeth_nums = data_load_no_centering(file_path)
        flags = 1
        file_path_split = file_path.split('/')
        data_name = file_path_split[-1][:-8]
        if 0 == flags:
            Rteeth_points = copy.deepcopy(Gteeth_points)
            mask_.append(ffi)
        else:
            Rteeth_points, teeth_nums_ = data_load_no_centering(file_path.replace("end.npy", "start.npy"))
            mask_.append(ffi)
        nums = cfg.teeth_nums
        rotate_nums = np.random.randint(0, nums, 1)[0]
        rotate_index = [i for i in range(nums)]
        np.random.shuffle(rotate_index)
        rotate_index = rotate_index[0: rotate_nums]
        Rweights = np.ones((Rteeth_points.shape[0]))
        Tweights = np.ones((Rteeth_points.shape[0]))
        rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
        if 0 == flags:
            for tid in rotate_index:
                rotaxis = np.random.random(3) *2 -1 + 0.01
                rotaxis =  rotaxis / np.linalg.norm(rotaxis)
                v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
                angle_ = v1 * cfg.Angles[np.random.randint(0, cfg.AgSize, 1)[0]]
                rt = rotate_maxtrix(rotaxis, angle_)
                rt = rt[0:3, 0:3]
                cen = np.mean(Rteeth_points[tid], axis=0)
                points = Rteeth_points[tid] - cen
                points_ = (rt.dot(points.T)).T
                Rteeth_points[tid] = points_ + cen
                rms[tid] = rt
                Rweights[tid] = Rweights[tid] + abs(angle_)*3 /100.0
            nums = len(teeth_nums)
            rotate_nums = np.random.randint(0, nums, 1)[0]
            rotate_index = [i for i in range(nums)]
            np.random.shuffle(rotate_index)
            rotate_index = rotate_index[0: rotate_nums]
            trans_v = np.array([[-2, -2, 2]])
            for i in range(Rteeth_points.shape[0]):
                index = np.random.randint(0, 3, 1)[0]
                rotaxis = cfg.ROTAXIS[index]
                v1 = np.random.normal(0, 1, size=(1))[0]
                v2 = np.random.normal(0, 1, size=(1))[0]
                v3 = np.random.normal(0, 1, size=(1))[0]
                fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)
                if i in rotate_index:
                    Rteeth_points[i] = Rteeth_points[i] + fg *trans_v
            Gcenp = np.mean(Gteeth_points, axis=1)
            Rcenp = np.mean(Rteeth_points, axis=1)
            trans = Transform3d().compose(Translate(torch.tensor(-Rcenp)),
                                          Rotate(torch.tensor(rms[:, 0:3, 0:3])),
                                          Translate(torch.tensor(Gcenp)))
            final_trans_mat = trans.get_matrix()
            gdofs = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            rcpoint = np.mean(Rteeth_points, axis=0)
            Rteeth_points = Rteeth_points - rcpoint
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            Gteeth_points = Gteeth_points - np.mean(Gteeth_points, axis=0)
            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
            trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float32)
            for di in range(Rteeth_points.shape[0]):
                censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
                trans_mats[di] = censd
                Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0
        else:
            mat_dir = '/devdata/dzx_data/tadpmData/singleMesh/train/' + data_name + '/toothMat.txt'
            with open(mat_dir, 'r', encoding='utf-8') as f:
                for ann in f.readlines():
                    ann = ann.strip('\n')
                    mat_nums = ann.split(' ')
                    tttid = mat_nums[0]
                    if tttid in cfg.INDEX:
                        index = int(cfg.INDEX[tttid]) - 1
                        mn = 1
                        for i in range(0, 4):
                            for j in range(0, 4):
                                if i != 3 and j != 3:
                                    rms[index][i][j] = mat_nums[mn]
                                mn = mn + 1
                        test_dofs = matrix_to_quaternion(torch.Tensor(rms[index]))
                        ax_an = quaternion_to_axis_angle(test_dofs)
                        angle_ = (ax_an[0] * ax_an[0] + ax_an[1] * ax_an[1] + ax_an[2] * ax_an[2]) **0.5
                        angle_ = angle_ / math.pi * 180.0
                        Rweights[index] = Rweights[index] + abs(angle_)*3 /100.0
            for tid in rotate_index:
                rotaxis = np.random.random(3) * 2 - 1 + 0.01
                rotaxis =  rotaxis / np.linalg.norm(rotaxis)
                v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
                avx = cfg.teeth_nums*0.0035
                angle_ = v1 * cfg.Angles[np.random.randint(0, cfg.AgSize, 1)[0]] * avx
                rt = rotate_maxtrix(rotaxis, angle_)
                rt = rt[0:3, 0:3]
                cen = np.mean(Rteeth_points[tid], axis=0)
                Rteeth_points[tid] = Rteeth_points[tid] - cen
                Rteeth_points[tid] = Rteeth_points[tid].dot(rt)
                Rteeth_points[tid] = Rteeth_points[tid] + cen
                rms[tid] = (rt.T).dot(rms[tid])
            nums_t = cfg.teeth_nums
            trans_nums = np.random.randint(0, nums_t, 1)[0]
            trans_index = [i for i in range(nums_t)]
            np.random.shuffle(trans_index)
            trans_index = trans_index[0: trans_nums]
            trans_v = np.array([[2, 2, 2]])
            cvx = nums_t*7e-4
            for tid in trans_index:
                v1 = np.random.normal(0, 1, size=(1))[0]
                v2 = np.random.normal(0, 1, size=(1))[0]
                v3 = np.random.normal(0, 1, size=(1))[0]
                fg = np.clip(np.array([[v1, v2, v3]]), -1, 1) * cvx
                Rteeth_points[tid] = Rteeth_points[tid] + fg * trans_v
            Gcenp = np.mean(Gteeth_points, axis=1)
            Rcenp = np.mean(Rteeth_points, axis=1)
            trans = Transform3d().compose(Translate(torch.tensor(-Rcenp)),
                                          Rotate(torch.tensor(rms[:, 0:3, 0:3])),
                                          Translate(torch.tensor(Gcenp)))
            final_trans_mat = trans.get_matrix()
            gdofs = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            rcpoint = np.mean(Rteeth_points, axis=0)
            Rteeth_points = Rteeth_points - rcpoint
            Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
            Gteeth_points = Gteeth_points - np.mean(Gteeth_points, axis=0)
            Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
            trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float32)
            for di in range(Rteeth_points.shape[0]):
                censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
                trans_mats[di] = censd
                Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0
        teeth_center = []
        for i in range(Rteeth_points.shape[0]):
            cenp = np.mean(Rteeth_points[i], axis=0)
            teeth_center.append(cenp)
        tGteeth_points.append(torch.tensor(np.array(Gteeth_points)))
        tRteeth_points.append(torch.tensor(np.array(Rteeth_points)))
        tteeth_center.append(torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1))
        rweights.append(torch.tensor(Rweights))
        tweights.append(torch.tensor(Tweights))
        tgdofs.append(gdofs)
        ttrans_mats.append(torch.tensor(trans_mats))
        rcpoints.append(rcpoint)
    tGteeth_points = torch.stack(tGteeth_points, dim=0)
    tRteeth_points = torch.stack(tRteeth_points, dim=0)
    tteeth_center = torch.stack(tteeth_center, dim=0)
    tweights = torch.stack(tweights, dim=0)
    rweights = torch.stack(rweights, dim=0)
    tgdofs = torch.stack(tgdofs, dim=0)
    ttrans_mats = torch.stack(ttrans_mats, dim=0)
    rcpoints = torch.tensor(np.array(rcpoints))
    mask_ = torch.tensor(np.array(mask_))       
    return tRteeth_points, tGteeth_points, tteeth_center, tgdofs, ttrans_mats, tweights, rweights, mask_