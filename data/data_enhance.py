import os
import sys
sys.path.append("/home/charon/codeGala/dzx/orth-tooth")
from config import config as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_divice_id
import json
import time
import torch
from torch import Tensor
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import *
import numpy as np
import vedo
import vtkmodules.all as vtk
from data.utils import walkFile, get_files, rotate_maxtrix
import shutil


def read_stl(file_path):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader

def write_stl(polydata, save_path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()

def get_polydata(triangles, rpoints):
    points = vtk.vtkPoints()
    for p in rpoints:
        points.InsertNextPoint(p[0], p[1], p[2])
    new_plyd = vtk.vtkPolyData()
    new_plyd.SetPoints(points)
    new_plyd.SetPolys(triangles)
    return new_plyd


def data_enhancement():
    file_path = "/devdata/dzx_data/tadpmData/singleMesh/train_stl_enhance"
    save_root = "/devdata/dzx_data/tadpmData/singleMesh/train_stl/"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    dir_list = []
    walkFile(file_path, dir_list)
    for fi in range(len(dir_list)):
        file_list = []
        file_p = dir_list[fi]
        print("\n", file_p)
        get_files(file_p, file_list, ".stl")
        dir_name = os.path.split(file_p)[-1]
        new_p = save_root + '/EH_' + dir_name
        if not os.path.exists(new_p):
            os.mkdir(new_p)
        if '_end' in dir_name:
            for di in range(len(file_list)):
                s_f_n = os.path.split(file_list[di])[-1]
                teeth_nums = s_f_n.replace(".stl", "").split("_")[-1]
                if teeth_nums not in cfg.INDEX.keys():
                    continue
                shutil.copy(file_list[di], new_p + '/EH_' + s_f_n)
        if '_start' in dir_name:
            nums = cfg.teeth_nums
            rotate_nums = np.random.randint(0, nums, 1)[0]
            rotate_index = [i for i in range(nums)]
            np.random.shuffle(rotate_index)
            rotate_index = rotate_index[0: rotate_nums]
            nums_t = cfg.teeth_nums
            trans_nums = np.random.randint(0, nums_t, 1)[0]
            trans_index = [i for i in range(nums_t)]
            np.random.shuffle(trans_index)
            trans_index = trans_index[0: trans_nums]
            trans_v = np.array([[2, -2, -2]])
            
            rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
            mat_dir = file_p + '/toothMat.txt'
            rms_after_eh = {}
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
            
            for di in range(len(file_list)):
                stl_reader = read_stl(file_list[di])
                teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
                polydata = stl_reader.GetOutput()
                points = polydata.GetPoints()
                verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
                r_points = np.array(verts)
                triangles = polydata.GetPolys()
                if teeth_nums not in cfg.INDEX.keys():
                    continue
                index = int(cfg.INDEX[teeth_nums]) - 1
                if index in rotate_index:
                    rotaxis = np.random.random(3) * 2 - 1 + 0.01
                    rotaxis =  rotaxis / np.linalg.norm(rotaxis)
                    v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
                    angle_ = v1 * cfg.Angles[np.random.randint(0, cfg.AgSize, 1)[0]]
                    rt = rotate_maxtrix(rotaxis, angle_)
                    rt = rt[0:3, 0:3]
                    cen = np.mean(r_points, axis=0)
                    r_points = r_points - cen
                    r_points = r_points.dot(rt)
                    r_points = r_points + cen
                    rms[index] = (rt.T).dot(rms[index])
                rms_after_eh[int(teeth_nums)] = rms[index]
                if index in trans_index:
                    v1 = np.random.normal(0, 1, size=(1))[0]
                    v2 = np.random.normal(0, 1, size=(1))[0]
                    v3 = np.random.normal(0, 1, size=(1))[0]
                    fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)
                    r_points = r_points + fg * trans_v
                after_polydata = get_polydata(triangles, r_points)
                s_f_n = os.path.split(file_list[di])[-1]
                write_stl(after_polydata, new_p + '/EH_' + s_f_n)
            
            rms_after_eh = sorted(rms_after_eh.items(), key=lambda v:v[0])
            rms_after_eh = dict(rms_after_eh)
            with open(new_p + '/toothMat.txt', 'w+', encoding='utf-8') as f:
                for key, value in rms_after_eh.items():
                    f.write(str(key) + " " + str(value[0][0]) + " " + str(value[0][1]) + " " + str(value[0][2]) + " " + "0" +
                                  " " + str(value[1][0]) + " " + str(value[1][1]) + " " + str(value[1][2]) + " " + "0" + 
                                  " " + str(value[2][0]) + " " + str(value[2][1]) + " " + str(value[2][2]) + " " + "0" + 
                                  " " + "0" + " " + "0" + " " + "0" + " " + "1" + "\n")

def simulate_end(dir_start):
    file_list = []
    get_files(dir_start, file_list, ".stl")

    rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
    mat_dir = dir_start + '/toothMat.txt'
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
    
    rappendFilter = vtk.vtkAppendPolyData()
    for di in range(len(file_list)):
        stl_reader = read_stl(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        if teeth_nums not in cfg.INDEX.keys():
            continue
        polydata = stl_reader.GetOutput()
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        r_points = np.array(verts)
        triangles = polydata.GetPolys()
        rcp = np.mean(r_points, axis=0)

        stl_reader_g = read_stl(file_list[di].replace("_start", "_end"))
        polydata_g = stl_reader_g.GetOutput()
        points_g = polydata_g.GetPoints()
        verts_g = np.array([points_g.GetPoint(i) for i in range(points_g.GetNumberOfPoints())])
        gpoints = np.array(verts_g)
        gcp = np.mean(gpoints, axis=0)

        index = int(cfg.INDEX[teeth_nums]) - 1
        r_points = r_points - rcp
        r_points = r_points.dot(rms[index])
        r_points = r_points + gcp

        rpolydata = get_polydata(triangles, r_points)
        rappendFilter.AddInputData(rpolydata)
    
    rappendFilter.Update()
    save_root =  dir_start + '/simulate_end.stl'
    write_stl(rappendFilter.GetOutput(), save_root)
    

if __name__ =="__main__":

    data_enhancement()