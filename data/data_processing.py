import os
import sys
sys.path.append("/home/charon/codeGala/dzx/orth-tooth")
from config import config as cfg
import json
import time
import numpy as np
import torch
import vedo
import vtkmodules.all as vtk
from data.utils import get_files, rotate_maxtrix
import functools
import random
import shutil

def get_polydata(triangles, rpoints):
    points = vtk.vtkPoints()
    for p in rpoints:
        points.InsertNextPoint(p[0], p[1], p[2])
    new_plyd = vtk.vtkPolyData()
    new_plyd.SetPoints(points)
    new_plyd.SetPolys(triangles)
    return new_plyd

def write_stl(polydata, save_path):
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()

INDEX = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,"8": 8,
         "9": 9, "10": 10, "11": 11, "12": 12, "13": 13,"14": 14, "15": 15, "16": 16,
         "17": 1, "18": 2, "19": 3, "20": 4, "21": 5, "22": 6,"23": 7, "24": 8,
         "25": 9, "26": 10, "27": 11, "28": 12, "29": 13,"30": 14, "31": 15, "32": 16}

INDEX_UP = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,"8": 8,
         "9": 9, "10": 10, "11": 11, "12": 12, "13": 13,"14": 14, "15": 15, "16": 16}
INDEX_DOWN = {"17": 1, "18": 2, "19": 3, "20": 4, "21": 5, "22": 6,"23": 7, "24": 8,
         "25": 9, "26": 10, "27": 11, "28": 12, "29": 13,"30": 14, "31": 15, "32": 16}

INDEXC = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

def read_stl(file_path):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader

def walkFile_t(path_root, file_list, type_):

    for root, dirs, files in os.walk(path_root):
        for d in dirs:
            path_file = os.path.join(root, d)
            if type_ in path_file:
                file_list.append(path_file)

def farthest_point_sample_GPU(xyz, npoint):
    device = xyz.device
    N, C = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N).to(device) * 1e10
    farthest = random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def farthest_point_sample(xyz, npoint):
    N, C = xyz.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = random.randint(0, N-1)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids

def compare_dis(a, b):
    if a["dis"] > b["dis"]:
        return 1
    elif a["dis"] < b["dis"]:
        return -1
    else:
        return 0

def train_data():
    T1 = time.time()
    file_path = "/devdata/dzx_data/tadpmData/singleMesh/train_stl"
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    dir_list = []
    walkFile_t(file_path, dir_list, "_end")

    save_root =  "/devdata/dzx_data/tadpmData/singleMesh/train/"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for fi in range(len(dir_list)):
        file_path = dir_list[fi]
        file_list = []
        print("\n", fi, "/", len(dir_list), file_path)
        dir_name = os.path.split(file_path)[-1]
        sub_dir_name = dir_name.replace("_end", "")
        sub_save_root = save_root + sub_dir_name
        if not os.path.exists(sub_save_root):
            os.mkdir(sub_save_root)
        save_points_path = sub_save_root + "/" + dir_name +".npy"
        save_points_path_start = sub_save_root + "/" + sub_dir_name + "_start.npy"
        mat_dir_ = file_path.replace("_end", "_start") + "/toothMat.txt"
        shutil.copy(mat_dir_, sub_save_root + "/toothMat.txt")
        if os.path.exists(save_points_path):
            continue
        if os.path.exists(save_points_path_start):
            continue
        get_files(file_path, file_list, ".stl")


        jaw_points = {}
        jaw_points_start = {}
        tooth_center = []
        for di in range(len(file_list)):
            stl_reader = read_stl(file_list[di])
            teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
            if teeth_nums not in INDEX.keys():
                continue
            polydata = stl_reader.GetOutput()
            for i in range(polydata.GetNumberOfPoints()):
                tooth_center.append(polydata.GetPoint(i))
        tooth_center = np.array(tooth_center)
        tooth_center = np.mean(tooth_center, axis=0)

            
        tkss = []
        for di in range(len(file_list)):
            corrs_start_dir = file_list[di].replace("_end", "_start")
            stl_reader = read_stl(file_list[di])
            stl_reader_start = read_stl(corrs_start_dir)

            teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
            if teeth_nums not in INDEX.keys():
                continue
            polydata = stl_reader.GetOutput()
            polydata_start = stl_reader_start.GetOutput()
            if polydata.GetNumberOfPoints() != polydata_start.GetNumberOfPoints():
                continue

            verts = []
            verts_start = []
            for i in range(polydata.GetNumberOfPoints()):
                verts.append(polydata.GetPoint(i))
            for i in range(polydata_start.GetNumberOfPoints()):
                verts_start.append(polydata_start.GetPoint(i))
            mesh_points = np.array(verts)
            mesh_points_start = np.array(verts_start)
            
            if mesh_points.shape[0] < cfg.sam_points:
                continue

            tkss.append(teeth_nums)
            fps_index = farthest_point_sample(mesh_points, cfg.sam_points)
            fps_index = np.array(fps_index)

            order_map = []
            order_ps = mesh_points[fps_index]
            for ix in range(len(order_ps)):
                item = {}
                if order_ps[ix][1] > tooth_center[1]:
                    temp_c = [tooth_center[0], order_ps[ix][1], tooth_center[2]]
                    item['dis'] = np.linalg.norm(temp_c - order_ps[ix])
                else:
                    item['dis'] = np.linalg.norm(tooth_center - order_ps[ix])
                item['idx'] = fps_index[ix]
                order_map.append(item)
            order_map.sort(key=functools.cmp_to_key(compare_dis))
            se_index_order = []
            for ix in range(len(order_map)):
                se_index_order.append(order_map[ix]['idx'])

            se_points = mesh_points[se_index_order]
            se_points_start = mesh_points_start[se_index_order]
            index = int(INDEX[teeth_nums])
            jaw_points[index] = se_points
            jaw_points_start[index] = se_points_start

        if file_path.find('up') != -1:
            for key, value in INDEX_UP.items():
                if key not in tkss:
                    index = int(value)
                    jaw_points[index] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
                    jaw_points_start[index] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
        else:
            for key, value in INDEX_DOWN.items():
                if key not in tkss:
                    index = int(value)
                    jaw_points[index] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
                    jaw_points_start[index] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
        
        np.save(save_points_path, jaw_points)
        np.save(save_points_path_start, jaw_points_start)
    T2 = time.time()
    print('dataProcessing: %s min' % ((T2 - T1) / 60))


if __name__ =="__main__":
    train_data()