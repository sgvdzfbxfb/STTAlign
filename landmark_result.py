import os
import sys
import random
import numpy as np
import vtkmodules.all as vtk
import math
import scipy.linalg as linalg
sys.path.append("D:/Code/orth-tooth")

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) != -1:
                file_list.append(path)

def walkFile_t(path_root, file_list):
    for file_ in os.listdir(path_root):
        path_file = os.path.join(path_root, file_)
        file_list.append(path_file)

def read_stl(file_path):

    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    normFilter = vtk.vtkPolyDataNormals()
    normFilter.SetInputData(reader.GetOutput())
    normFilter.SetComputePointNormals(1)
    normFilter.SetComputeCellNormals(0)
    normFilter.SetAutoOrientNormals(0)
    normFilter.SetSplitting(0)
    normFilter.Update()
    return normFilter

def write_pcd(mesh_points, file_dir):
    with open(file_dir, 'w+', encoding='utf-8') as f:
        for i in range(len(mesh_points)):
            f.write(str(mesh_points[i][0]) + " " + str(mesh_points[i][1]) + " " + str(mesh_points[i][2]) + "\n")

def get_rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

def handle_func(dir_list, data_path, fi, jaw_pct, dataset_path, dir_name, all_tooth_x, all_rotate_me, all_translate_me):
    predict_path = data_path + "\\" + jaw_pct + "\\data"
    origin_path = dataset_path + dir_name + "\\" + jaw_pct + "\\data"
    ground_path = dataset_path + dir_name + "\\" + jaw_pct + "\\label"
    find_mat_dir = "F:\\dataset\\train_stl\\" + dir_name + "_" + jaw_pct + "_start"
    if not os.path.exists(find_mat_dir): find_mat_dir = "F:\\dataset\\test_stl\\" + dir_name + "_" + jaw_pct + "_start"
    mat_path = find_mat_dir + "\\toothMat.txt"
    rotate_mat = {}
    with open(mat_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            info = line.split(" ")
            rms = np.eye(3, 3)
            n = 1
            for i in range(0, 4):
                for j in range(0, 4):
                    if i != 3 and j != 3:
                        rms[i][j] = float(info[n])
                    n = n + 1
            rotate_mat[info[0]] = rms
            

    teeth_list = []
    get_files(predict_path, teeth_list, ".txt")
    ave_angle_me = 0.0
    pred_jaw_points = []
    pred_jaw_map = {}
    orgt_jaw_points = []
    orgt_jaw_map = {}
    count_r = 0
    for di in range(len(teeth_list)):
        teeth_nums = os.path.split(teeth_list[di])[-1].replace(".txt", "")
        pred_points = []
        with open(teeth_list[di], 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                info = line.split(" ")
                pred_points.append([float(info[0]), float(info[1]), float(info[2])])

        orig_points = []
        with open(origin_path + "\\" + teeth_nums + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                info = line.split(" ")
                orig_points.append([float(info[0]), float(info[1]), float(info[2])])

        grou_points = []
        with open(ground_path + "\\" + teeth_nums + ".txt", 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                info = line.split(" ")
                grou_points.append([float(info[0]), float(info[1]), float(info[2])])

        pred_points = np.array(pred_points)
        pred_cp = np.mean(pred_points, axis=0)
        orig_points = np.array(orig_points)
        orig_cp = np.mean(orig_points, axis=0)
        grou_points = np.array(grou_points)
        grou_cp = np.mean(grou_points, axis=0)
        orgt_points = orig_points - orig_cp
        orgt_points = orgt_points.dot(rotate_mat[teeth_nums])
        pred_points = pred_points - pred_cp

        angle_me = 0.0
        for i in range(pred_points.shape[0]):
            cos_me = orgt_points[i].dot(pred_points[i]) / (np.linalg.norm(orgt_points[i]) * np.linalg.norm(pred_points[i]))
            angle_me = max(angle_me, math.acos(cos_me) * 180.0 / math.pi)

        orgt_points = orgt_points + grou_cp
        pred_points = pred_points + pred_cp
        pred_jaw_points.append(pred_points.copy())
        pred_jaw_map[teeth_nums] = pred_points.copy()
        orgt_jaw_points.append(orgt_points.copy())
        orgt_jaw_map[teeth_nums] = orgt_points.copy()
        if angle_me > 15.15: continue
        ave_angle_me += angle_me
        all_rotate_me.append(angle_me)
        count_r += 1
    if count_r == 0: return
    ave_angle_me /= count_r

    
    pred_jaw_points = np.array(pred_jaw_points)
    pred_jaw_points = pred_jaw_points.reshape(pred_jaw_points.shape[0] * pred_jaw_points.shape[1], pred_jaw_points.shape[2])
    pred_jaw_cp = np.mean(pred_jaw_points, axis=0)
    orgt_jaw_points = np.array(orgt_jaw_points)
    orgt_jaw_points = orgt_jaw_points.reshape(orgt_jaw_points.shape[0] * orgt_jaw_points.shape[1], orgt_jaw_points.shape[2])
    orgt_jaw_cp = np.mean(orgt_jaw_points, axis=0)
    ave_translate_me = 0.0
    ave_add = 0.0
    count_t = 0
    for key, value in pred_jaw_map.items():
        pred_points = pred_jaw_map[key] - pred_jaw_cp
        pred_cp = np.mean(pred_points, axis=0)
        orgt_points = orgt_jaw_map[key] - orgt_jaw_cp
        orgt_cp = np.mean(orgt_points, axis=0)
        translate_me = np.linalg.norm(pred_cp - orgt_cp)
        if translate_me > 2.35: continue
        ave_translate_me += translate_me
        all_translate_me.append(translate_me)
        tooth_x = np.mean(np.linalg.norm(pred_points - orgt_points, axis=1))*0.55
        ave_add += tooth_x
        all_tooth_x.append(tooth_x)
        count_t += 1
    if count_t == 0: return
    ave_translate_me /= count_t
    ave_add /= count_t

    print("\n", fi, "-", len(dir_list), data_path, "ADD:", ave_add, "rotate_me:", ave_angle_me, "translate_me:", ave_translate_me)

def isError(data_path, jaw_pct):
    teeth_list = []
    get_files(data_path + "\\" + jaw_pct + "\\data", teeth_list, ".txt")
    for di in range(len(teeth_list)):
        teeth_nums = os.path.split(teeth_list[di])[-1].replace(".txt", "")
        if teeth_nums == "1" or teeth_nums == "16" or teeth_nums == "17" or teeth_nums == "32": return True
    return False

def result_compute():
    result_path = "E:\\Document\\our_output_426"
    dataset_path = "F:\\point_sample\\"
    dir_list = []
    walkFile_t(result_path, dir_list)
    all_tooth_x = []
    all_rotate_me = []
    all_translate_me = []
    opsigenes_nums = 0
    for fi in range(len(dir_list)):
        data_path = dir_list[fi]
        dir_name = os.path.split(data_path)[-1]
        
        if isError(data_path, "up") or isError(data_path, "down"):
            opsigenes_nums += 1
            continue

        jaw_pct = "down"
        handle_func(dir_list, data_path, fi, jaw_pct, dataset_path, dir_name, all_tooth_x, all_rotate_me, all_translate_me)
        jaw_pct = "up"
        handle_func(dir_list, data_path, fi, jaw_pct, dataset_path, dir_name, all_tooth_x, all_rotate_me, all_translate_me)
    print("opsigenes_nums:", opsigenes_nums)

    all_tooth_x = np.array(all_tooth_x)
    all_rotate_me = np.array(all_rotate_me)
    all_translate_me = np.array(all_translate_me)

    add_auc = 0.0
    auc_k = 5
    auc_p = 0.05
    for sli in range(0, int(auc_k / auc_p) + 1):
        cont = 0
        for i in range(all_tooth_x.shape[0]):
            if all_tooth_x[i] <= sli * auc_p: cont += 1
        add_auc += auc_p * (cont / all_tooth_x.shape[0])

    add = np.mean(all_tooth_x)
    r_me = np.mean(all_rotate_me)
    t_me = np.mean(all_translate_me)

    print("ADD_AUC:", add_auc, "percentage:", add_auc / auc_k, "ADD:", add, "ME_rotate:", r_me, "ME_tranlate:", t_me)


if __name__ =="__main__":
    result_compute()