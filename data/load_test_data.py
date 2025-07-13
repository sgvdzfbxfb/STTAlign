import os
import math
import copy
import torch
from pytorch3d.structures import Meshes, Pointclouds
import vedo
import numpy as np
import vtkmodules.all as vtk
from data.utils import get_files, walkFile
from data.utils import rotate_maxtrix
import config.config as cfg
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle,Transform3d,Rotate,quaternion_to_matrix
import random
import functools

def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader

def compare_dis(a, b):
    if a["dis"] > b["dis"]:
        return 1
    elif a["dis"] < b["dis"]:
        return -1
    else:
        return 0

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


def load_stl_data_to_code(file_list):
    teeth_points = np.zeros((cfg.teeth_nums, cfg.sam_points, 3), np.float64)
    teeth_points_center = np.zeros((cfg.teeth_nums, 3), np.float64)
    se_ind = {}
    tooth_center = []
    for di in range(len(file_list)):
        if not os.path.exists(file_list[di]):
            continue
        stl_reader = read_stl(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        if teeth_nums not in cfg.INDEX.keys():
            continue
        polydata = stl_reader.GetOutput()
        for i in range(polydata.GetNumberOfPoints()):
            tooth_center.append(polydata.GetPoint(i))
    tooth_center = np.array(tooth_center)
    tooth_center = np.mean(tooth_center, axis=0)
    tkss = []
    for di in range(len(file_list)):
        if not os.path.exists(file_list[di]):
            continue
        stl_reader = read_stl(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        if teeth_nums not in cfg.INDEX.keys():
            continue
        polydata = stl_reader.GetOutput()
        if not os.path.exists(file_list[di].replace("_end", "_start")):
            continue
        start_stl_reader = read_stl(file_list[di].replace("_end", "_start"))
        start_polydata = start_stl_reader.GetOutput()
        if polydata.GetNumberOfPoints() != start_polydata.GetNumberOfPoints():
            continue
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        if mesh_points.shape[0] < cfg.sam_points:
            continue
        tkss.append(teeth_nums)
        se_index = farthest_point_sample(mesh_points, cfg.sam_points)
        se_index = np.array(se_index)
        order_map = []
        order_ps = mesh_points[se_index]
        for ix in range(len(order_ps)):
            item = {}
            if order_ps[ix][1] > tooth_center[1]:
                temp_c = [tooth_center[0], order_ps[ix][1], tooth_center[2]]
                item['dis'] = np.linalg.norm(temp_c - order_ps[ix])
            else:
                item['dis'] = np.linalg.norm(tooth_center - order_ps[ix])
            item['idx'] = se_index[ix]
            order_map.append(item)
        order_map.sort(key=functools.cmp_to_key(compare_dis))
        se_index_order = []
        for ix in range(len(order_map)):
            se_index_order.append(order_map[ix]['idx'])
        se_ind[teeth_nums] = se_index_order
        se_points = mesh_points[se_index_order]
        index = int(cfg.INDEX[teeth_nums])
        teeth_points[index - 1] = se_points
        teeth_points_center[index - 1] = np.mean(mesh_points, axis=0)
    if file_list[0].find('up') != -1:
        for key, value in cfg.INDEX_UP.items():
            if key not in tkss:
                index = int(value)
                teeth_points[index - 1] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
    else:
        for key, value in cfg.INDEX_DOWN.items():
            if key not in tkss:
                index = int(value)
                teeth_points[index - 1] = np.expand_dims(tooth_center, axis=0).repeat(cfg.sam_points, axis=0)
    return teeth_points, teeth_points_center, se_ind, tooth_center

def load_stl_data_to_code_cuple(file_list, se_ind, tcs):
    teeth_points = np.zeros((cfg.teeth_nums, cfg.sam_points, 3), np.float64)
    teeth_points_center = np.zeros((cfg.teeth_nums, 3), np.float64)
    tkss = []
    for di in range(len(file_list)):
        if not os.path.exists(file_list[di]):
            continue
        stl_reader = read_stl(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        if teeth_nums not in cfg.INDEX.keys() or teeth_nums not in se_ind.keys():
            continue
        polydata = stl_reader.GetOutput()
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        if mesh_points.shape[0] < cfg.sam_points:
            continue
        tkss.append(teeth_nums)
        se_points = mesh_points[se_ind[teeth_nums]]
        index = int(cfg.INDEX[teeth_nums])
        teeth_points[index - 1] = se_points
        teeth_points_center[index - 1] = np.mean(mesh_points, axis=0)
    if file_list[0].find('up') != -1:
        for key, value in cfg.INDEX_UP.items():
            if key not in tkss:
                index = int(value)
                teeth_points[index - 1] = np.expand_dims(tcs, axis=0).repeat(cfg.sam_points, axis=0)
    else:
        for key, value in cfg.INDEX_DOWN.items():
            if key not in tkss:
                index = int(value)
                teeth_points[index - 1] = np.expand_dims(tcs, axis=0).repeat(cfg.sam_points, axis=0)
    return teeth_points, teeth_points_center

def get_test_data(file_path_start, file_list, file_list_start):
    teeth_points, Gcps, se_ind, tcs = load_stl_data_to_code(file_list)
    teeth_points_start, Rcps = load_stl_data_to_code_cuple(file_list_start, se_ind, tcs)
    Gteeth_points = copy.deepcopy(teeth_points).reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    Gacenp = np.mean(Gteeth_points, axis=0, keepdims=True)
    Gteeth_points = Gteeth_points - Gacenp
    Gteeth_points = Gteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
    Rteeth_points = copy.deepcopy(teeth_points_start).reshape(cfg.teeth_nums * cfg.sam_points, cfg.dim)
    Rcp = np.mean(Rteeth_points, axis=0, keepdims=True)
    Rteeth_points = Rteeth_points - Rcp
    Rteeth_points = Rteeth_points.reshape(cfg.teeth_nums, cfg.sam_points, cfg.dim)
    Rweights = np.ones((Rteeth_points.shape[0]))
    Tweights = np.ones((Rteeth_points.shape[0]))
    rms = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
    mat_dir = file_path_start + '/toothMat.txt'
    with open(mat_dir, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            mat_nums = ann.split(' ')
            tttid = mat_nums[0]
            if tttid in cfg.INDEX:
                index = int(cfg.INDEX[tttid]) -1
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
                Rweights[index] = Rweights[index] + abs(angle_) * 3 /100.0
    trans = Transform3d().compose(Rotate(torch.tensor(rms[:, 0:3, 0:3])))
    final_trans_mat = trans.get_matrix()
    dof = matrix_to_quaternion(final_trans_mat[:, 0:3, 0:3])
    trans_mats = np.zeros((Rteeth_points.shape[0], 3), np.float64)
    for di in range(Rteeth_points.shape[0]):
        censd = np.mean(Gteeth_points[di], axis=0) - np.mean(Rteeth_points[di], axis=0)
        trans_mats[di] = censd
        Tweights[di] = Tweights[di] + abs(np.sum(censd)) / 10.0
    teeth_center = []
    Rmove_bias = []
    for i in range(Rteeth_points.shape[0]):
        cenp = np.mean(Rteeth_points[i], axis=0)
        Rmove_bias.append(cenp - Rcps[i])
        teeth_center.append(cenp)
    Gmove_bias = []
    for i in range(Gteeth_points.shape[0]):
        cenp = np.mean(Gteeth_points[i], axis=0)
        Gmove_bias.append(cenp - Gcps[i])
    Gteeth_points = torch.unsqueeze(torch.tensor(np.array(Gteeth_points)), dim=0)
    Rteeth_points = torch.unsqueeze(torch.tensor(np.array(Rteeth_points)), dim=0)
    Rweights = torch.unsqueeze(torch.tensor(np.array(Rweights)), dim=0)
    Tweights = torch.unsqueeze(torch.tensor(np.array(Tweights)), dim=0)
    trans_mats = torch.Tensor(trans_mats)
    cfg.decoder_r = np.array(dof)
    cfg.decoder_t = np.array(trans_mats)
    teeth_center = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(teeth_center)), dim=1), dim=0)
    Rmove_bias = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(Rmove_bias)), dim=1), dim=0)
    Gmove_bias = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array(Gmove_bias)), dim=1), dim=0)
    return  Rteeth_points, Gteeth_points, teeth_center, Rweights, Tweights, rms, dof, trans_mats, Gmove_bias, Rmove_bias

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

def compare(a, b):
    if int(a["key"]) > int(b["key"]):
        return 1
    elif int(a["key"]) < int(b["key"]):
        return -1
    else:
        return 0

def mapping_output(where_write, is_the_last, file_list, pdofs, ptrans, Gmove_bias, Rmove_bias, save_path, save_name):
    pred_matrices = torch.cat([quaternion_to_matrix(pdofs[idx]).unsqueeze(0) for idx in range(pdofs.shape[0])], dim = 0)
    pred_matrices = torch.squeeze(pred_matrices).detach().cpu().numpy()
    ptrans = torch.squeeze(ptrans).detach().cpu().numpy()
    gappendFilter = vtk.vtkAppendPolyData()
    rappendFilter = vtk.vtkAppendPolyData()
    rvappendFilter = vtk.vtkAppendPolyData()
    ths_tooth_x = []
    shujuji_dir_en = os.path.split(file_list[0])[0].split('/')[-1]
    shujuji_dir_en = where_write + "/" + shujuji_dir_en
    shujuji_dir_st = shujuji_dir_en.replace("end", "start")
    if not is_the_last:
        if not os.path.exists(shujuji_dir_en):
            os.mkdir(shujuji_dir_en)
        if not os.path.exists(shujuji_dir_st):
            os.mkdir(shujuji_dir_st)
    p_args = os.path.split(file_list[0])[0:-1]
    lia_dir = ""
    for sd in range(len(p_args)):
        lia_dir = lia_dir + p_args[sd] + "/"
    lia_dir = lia_dir.replace("_end", "_start")
    mat_dir = lia_dir + "toothMat.txt"
    rmss = np.eye(3, 3).reshape(1, 3, 3).repeat(cfg.teeth_nums, axis=0)
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
                            rmss[index][i][j] = mat_nums[mn]
                        mn = mn + 1
    rms_new = []
    for di in range(len(file_list)):
        if not os.path.exists(file_list[di]):
            continue
        stl_reader = read_stl(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
        if teeth_nums not in cfg.INDEX.keys():
            continue
        index = int(cfg.INDEX[teeth_nums]) - 1
        polydata = stl_reader.GetOutput()
        points = polydata.GetPoints()
        verts = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        mesh_points = np.array(verts)
        triangles = polydata.GetPolys()
        if not os.path.exists(file_list[di].replace("_end", "_start")):
            continue
        stl_reader_r = read_stl(file_list[di].replace("_end", "_start"))
        polydata_r = stl_reader_r.GetOutput()
        points_r = polydata_r.GetPoints()
        verts_r = np.array([points_r.GetPoint(i) for i in range(points_r.GetNumberOfPoints())])
        rpoints = np.array(verts_r)
        triangles_r = polydata_r.GetPolys()
        if points.GetNumberOfPoints() != points_r.GetNumberOfPoints():
            continue
        rvpoints = rpoints
        rcp = np.mean(rvpoints, axis=0)
        rvpoints = rvpoints - rcp
        rvpoints = rvpoints.dot(pred_matrices[index])
        rvpoints = rvpoints + np.array(Rmove_bias.squeeze())[index] + ptrans[index] - np.array(Gmove_bias.squeeze())[index]
        rvpoints = rvpoints + rcp
        tooth_x = torch.mean(torch.norm(torch.Tensor(mesh_points - rvpoints), dim=1))
        ths_tooth_x.append(tooth_x)
        gpolydata = get_polydata(triangles, mesh_points)
        rpolydata = get_polydata(triangles_r, rpoints)
        rvpolydata = get_polydata(triangles_r, rvpoints)
        mat_rv_2_g = pred_matrices[index].T @ rmss[index]
        save_to_new_dir_en = shujuji_dir_en + "/" + os.path.split(file_list[di])[-1]
        save_to_new_dir_st = save_to_new_dir_en.replace("end", "start")
        if not is_the_last:
            write_stl(gpolydata, save_to_new_dir_en)
            write_stl(rvpolydata, save_to_new_dir_st)
            rms_new.append({"key": teeth_nums, "value": np.array(mat_rv_2_g)})
        if is_the_last:
            gappendFilter.AddInputData(gpolydata)
            rappendFilter.AddInputData(rpolydata)
            rvappendFilter.AddInputData(rvpolydata)
    if not is_the_last:
        rms_new.sort(key=functools.cmp_to_key(compare))
        new_mat_dir = shujuji_dir_st + "/" + "toothMat.txt"
        with open(new_mat_dir, 'w+', encoding='utf-8') as f:
            for i in range(len(rms_new)):
                f.write(rms_new[i]["key"] + 
                    " " + str(rms_new[i]["value"][0][0]) + " " + str(rms_new[i]["value"][0][1]) + " " + str(rms_new[i]["value"][0][2]) + " " + "0" +
                    " " + str(rms_new[i]["value"][1][0]) + " " + str(rms_new[i]["value"][1][1]) + " " + str(rms_new[i]["value"][1][2]) + " " + "0" + 
                    " " + str(rms_new[i]["value"][2][0]) + " " + str(rms_new[i]["value"][2][1]) + " " + str(rms_new[i]["value"][2][2]) + " " + "0" + 
                    " " + "0" + " " + "0" + " " + "0" + " " + "1" + "\n")
    if is_the_last:
        gappendFilter.Update()
        rappendFilter.Update()
        rvappendFilter.Update()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        write_stl(gappendFilter.GetOutput(), save_path + "/" + save_name + "_g"  + ".stl")
        write_stl(rappendFilter.GetOutput(), save_path + "/" + save_name + "_r"  + ".stl")
        write_stl(rvappendFilter.GetOutput(), save_path + "/" + save_name + "_rv" + ".stl")
    return ths_tooth_x