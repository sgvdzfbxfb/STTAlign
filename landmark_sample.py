import os
import sys
import random
import numpy as np
import vtkmodules.all as vtk
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

    for root, dirs, files in os.walk(path_root):
        for d in dirs:
            path_file = os.path.join(root, d)
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

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
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


def sample_func(sam_points, all_sampoints):
    file_path = "F:/dataset/test_stl"
    if not os.path.exists(file_path): os.mkdir(file_path)
    save_root =  "F:/point_sample/"
    if not os.path.exists(save_root): os.mkdir(save_root)
    dir_list = []
    walkFile_t(file_path, dir_list)
    for fi in range(len(dir_list)):
        file_path = dir_list[fi]
        file_list = []
        print("\n", fi, "/", len(dir_list), file_path)
        dir_name = os.path.split(file_path)[-1]
        jaw_sit = ""
        spc_sit = ""
        if "_up" in dir_name:jaw_sit = "up"
        elif "_down" in dir_name: jaw_sit = "down"
        if "_start" in dir_name: spc_sit = "start"
        elif "_end" in dir_name: spc_sit = "end"
        data_name = dir_name.replace("_" + jaw_sit + "_" + spc_sit, "")
        data_dir = save_root + data_name
        if not os.path.exists(data_dir): os.mkdir(data_dir)
        sub_dir1 = data_dir + "/" + jaw_sit
        if not os.path.exists(sub_dir1): os.mkdir(sub_dir1)
        ddr = "data" if spc_sit == "start" else "label"
        sub_dir2 = sub_dir1 + "/" + ddr
        if not os.path.exists(sub_dir2): os.mkdir(sub_dir2)


        all_mesh_point = np.zeros(1, dtype=float)
        all_mesh_normal = np.zeros(1, dtype=float)
        get_files(file_path, file_list, ".stl")
        for di in range(len(file_list)):
            stl_reader = read_stl(file_list[di])
            teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[-1]
            polydata = stl_reader.GetOutput()
            verts = []
            norms = []
            normdata = polydata.GetPointData().GetNormals()
            for i in range(polydata.GetNumberOfPoints()):
                verts.append(polydata.GetPoint(i))
                nn = [0, 0, 0]
                normdata.GetTuple(i, nn)
                norms.append(nn)
            mesh_points = np.array(verts)
            mesh_normals = np.array(norms)
            if mesh_points.shape[0] < sam_points: continue
            fps_index = farthest_point_sample(mesh_points, sam_points)
            fps_index = np.array(fps_index)
            fps_parr = mesh_points[fps_index]
            fps_narr = mesh_normals[fps_index]

            if di == 0:
                all_mesh_point = mesh_points.copy()
                all_mesh_normal = mesh_normals.copy()
            else:
                all_mesh_point = np.append(all_mesh_point, mesh_points, axis=0)
                all_mesh_normal = np.append(all_mesh_normal, mesh_normals, axis=0)

            with open(sub_dir2 + "/" + teeth_nums + ".txt", 'w+', encoding='utf-8') as f:
                for i in range(len(fps_parr)):
                    f.write(str(fps_parr[i][0]) + " " + str(fps_parr[i][1]) + " " + str(fps_parr[i][2]) + " " + str(fps_narr[i][0]) + " " + str(fps_narr[i][1]) + " " + str(fps_narr[i][2]) + "\n")
        a_fps_index = farthest_point_sample(all_mesh_point, all_sampoints)
        a_fps_index = np.array(a_fps_index)
        fps_all = all_mesh_point[a_fps_index]
        fps_norm = all_mesh_normal[a_fps_index]
        with open(sub_dir2 + "/" + jaw_sit + "_jaw.txt", 'w+', encoding='utf-8') as f:
            for i in range(len(fps_all)):
                f.write(str(fps_all[i][0]) + " " + str(fps_all[i][1]) + " " + str(fps_all[i][2]) + " " + str(fps_norm[i][0]) + " " + str(fps_norm[i][1]) + " " + str(fps_norm[i][2]) + "\n")
        

if __name__ =="__main__":
    sample_func(400, 2048)