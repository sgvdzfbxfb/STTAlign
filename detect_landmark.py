import numpy as np
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import vtkmodules.all as vtk
import sys

def load_stl_file(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    poly_data = reader.GetOutput()
    points = np.array([poly_data.GetPoint(i) for i in range(poly_data.GetNumberOfPoints())])
    return points

def load_tooth_info(file_path):
    tooth_info = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 13:
                tooth_num = int(data[0])
                center = np.array([float(x) for x in data[1:4]])
                x_axis = np.array([float(x) for x in data[4:7]])
                y_axis = np.array([float(x) for x in data[7:10]])
                z_axis = np.array([float(x) for x in data[10:13]])
                tooth_info[tooth_num] = [center, x_axis, y_axis, z_axis]
    return tooth_info

def determine_tooth_type(tooth_num):
    if 1 <= tooth_num <= 16:
        if tooth_num in [1, 16]:
            return 'molar'
        elif tooth_num in [2, 3, 14, 15]:
            return 'molar'
        elif tooth_num in [4, 5, 12, 13]:
            return 'premolar'
        elif tooth_num in [6, 11]:
            return 'canine'
        elif 7 <= tooth_num <= 10:
            return 'incisor'
    else:
        if tooth_num in [17, 32]:
            return 'molar'
        elif tooth_num in [18, 19, 30, 31]:
            return 'molar'
        elif tooth_num in [20, 21, 28, 29]:
            return 'premolar'
        elif tooth_num in [22, 27]:
            return 'canine'
        elif 23 <= tooth_num <= 26:
            return 'incisor'
    return 'unknown'

def compute_curvature(points: np.ndarray, k_neighbors: int = 30):
    tree = KDTree(points)
    N = points.shape[0]
    gaussian_curvature = np.zeros(N)
    mean_curvature = np.zeros(N)
    principal_curvatures = np.zeros((N, 2))
    for i in range(N):
        distances, indices = tree.query(points[i], k=k_neighbors)
        neighbors = points[indices]
        centroid = np.mean(neighbors, axis=0)
        centered = neighbors - centroid
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        normal = eigenvectors[:, 0]
        if eigenvalues[0] > 1e-8:
            k1 = eigenvalues[2] / eigenvalues[0]
            k2 = eigenvalues[1] / eigenvalues[0]
            principal_curvatures[i] = [k1, k2]
            gaussian_curvature[i] = k1 * k2
            mean_curvature[i] = (k1 + k2) / 2
    return gaussian_curvature, mean_curvature, principal_curvatures

def extract_tooth_landmarks_with_curvature(tooth_points: np.ndarray, tooth_type: str, tooth_axes: list):
    gaussian_curv, mean_curv, principal_curvs = compute_curvature(tooth_points)
    origin = tooth_axes[0]
    x_axis = tooth_axes[1]
    y_axis = tooth_axes[2]
    z_axis = tooth_axes[3]
    transform_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    local_points = (tooth_points - origin) @ transform_matrix
    landmarks = {}
    if tooth_type == 'incisor' or tooth_type == 'canine':
        high_points = local_points[local_points[:, 2] > np.percentile(local_points[:, 2], 90)]
        high_curv = gaussian_curv[local_points[:, 2] > np.percentile(local_points[:, 2], 90)]
        if len(high_points) > 0:
            incisal_point_local = high_points[np.argmax(high_curv)]
            incisal_point = incisal_point_local @ transform_matrix.T + origin
            landmarks['incisal_point'] = incisal_point
    elif tooth_type in ['premolar', 'molar']:
        z_range = np.percentile(local_points[:, 2], [60, 80])
        occlusal_mask = (local_points[:, 2] >= z_range[0]) & (local_points[:, 2] <= z_range[1])
        occlusal_points = local_points[occlusal_mask]
        occlusal_gaussian_curv = gaussian_curv[occlusal_mask]
        buccal_mask = occlusal_points[:, 0] > 0
        lingual_mask = occlusal_points[:, 0] <= 0
        if np.any(buccal_mask):
            buccal_points = occlusal_points[buccal_mask]
            buccal_curv = occlusal_gaussian_curv[buccal_mask]
            mesial_mask = buccal_points[:, 2] < np.median(buccal_points[:, 2])
            distal_mask = ~mesial_mask
            if np.any(mesial_mask):
                buccal_mesial_local = buccal_points[mesial_mask][np.argmax(buccal_curv[mesial_mask])]
                landmarks['buccal_mesial_cusp'] = buccal_mesial_local @ transform_matrix.T + origin
            if np.any(distal_mask):
                buccal_distal_local = buccal_points[distal_mask][np.argmax(buccal_curv[distal_mask])]
                landmarks['buccal_distal_cusp'] = buccal_distal_local @ transform_matrix.T + origin
        if np.any(lingual_mask):
            lingual_points = occlusal_points[lingual_mask]
            lingual_curv = occlusal_gaussian_curv[lingual_mask]
            mesial_mask = lingual_points[:, 2] < np.median(lingual_points[:, 2])
            distal_mask = ~mesial_mask
            if np.any(mesial_mask):
                lingual_mesial_local = lingual_points[mesial_mask][np.argmax(lingual_curv[mesial_mask])]
                landmarks['lingual_mesial_cusp'] = lingual_mesial_local @ transform_matrix.T + origin
            if np.any(distal_mask):
                lingual_distal_local = lingual_points[distal_mask][np.argmax(lingual_curv[distal_mask])]
                landmarks['lingual_distal_cusp'] = lingual_distal_local @ transform_matrix.T + origin
        central_region_mask = (np.abs(occlusal_points[:, 0]) < np.std(occlusal_points[:, 0])) & \
                            (np.abs(occlusal_points[:, 2]) < np.std(occlusal_points[:, 2]))
        if np.any(central_region_mask):
            central_points = occlusal_points[central_region_mask]
            central_curv = occlusal_gaussian_curv[central_region_mask]
            central_fossa_local = central_points[np.argmin(central_curv)]
            landmarks['central_fossa'] = central_fossa_local @ transform_matrix.T + origin
    return landmarks, {
        'gaussian_curvature': gaussian_curv,
        'mean_curvature': mean_curv,
        'principal_curvatures': principal_curvs
    }

def visualize_curvature_and_landmarks(tooth_points, curvature_data, landmarks):
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(tooth_points[:, 0], tooth_points[:, 2], tooth_points[:, 1],
                         c=curvature_data['gaussian_curvature'],
                         cmap='coolwarm')
    plt.colorbar(scatter, ax=ax1)
    ax1.set_title('高斯曲率')
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(tooth_points[:, 0], tooth_points[:, 2], tooth_points[:, 1],
                         c=curvature_data['mean_curvature'],
                         cmap='coolwarm')
    plt.colorbar(scatter, ax=ax2)
    ax2.set_title('平均曲率')
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(tooth_points[:, 0], tooth_points[:, 2], tooth_points[:, 1],
                c='gray', alpha=0.1, s=1)
    colors = {'incisal_point': 'red',
              'buccal_mesial_cusp': 'blue',
              'buccal_distal_cusp': 'green',
              'lingual_mesial_cusp': 'yellow',
              'lingual_distal_cusp': 'purple',
              'central_fossa': 'black'}
    for name, point in landmarks.items():
        if name in colors:
            ax3.scatter(point[0], point[2], point[1],
                       c=colors[name], s=100, label=name)
    ax3.legend()
    ax3.set_title('特征点')
    plt.tight_layout()
    plt.show()

def take_tooth_coordinate_system(tooth_info, tooth_num):
    if tooth_num not in tooth_info:
        raise ValueError(f"牙齿编号 {tooth_num} 在tooth_info中未找到")
    origin = tooth_info[tooth_num][0]
    axes = np.array([
        tooth_info[tooth_num][1],
        tooth_info[tooth_num][2],
        tooth_info[tooth_num][3]
    ])
    return origin, axes

def plot_coordinate_system(ax, origin, axes, scale=5.0):
    colors = ['r', 'g', 'b']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        direction = axes[i] * scale
        ax.quiver(origin[0], origin[1], origin[2],
                 direction[0], direction[1], direction[2],
                 color=colors[i], label=labels[i])

def test_landmark_detection(stl_folder_path, tooth_info_path):
    if not os.path.exists(stl_folder_path):
        print(f"错误：STL文件夹路径 '{stl_folder_path}' 不存在")
        return
    if not os.path.exists(tooth_info_path):
        print(f"错误：牙齿信息文件 '{tooth_info_path}' 不存在")
        return
    try:
        tooth_info = load_tooth_info(tooth_info_path)
        print(f"成功加载牙齿信息文件")
        stl_files = [f for f in os.listdir(stl_folder_path) if f.lower().endswith('.stl')]
        if not stl_files:
            print(f"在 {stl_folder_path} 中没有找到STL文件")
            return
        print(f"找到 {len(stl_files)} 个STL文件")
        for stl_file in stl_files:
            try:
                print(f"\n处理文件: {stl_file}")
                tooth_num = int(''.join(filter(str.isdigit, stl_file)))
                print(f"识别到牙齿编号: {tooth_num}")
                tooth_type = determine_tooth_type(tooth_num)
                print(f"牙齿类型: {tooth_type}")
                full_path = os.path.join(stl_folder_path, stl_file)
                tooth_points = load_stl_file(full_path)
                print(f"成功加载STL文件，包含 {len(tooth_points)} 个点")
                origin, axes = take_tooth_coordinate_system(tooth_info, tooth_num)
                tooth_axes = [origin, axes[0], axes[1], axes[2]]
                print("成功计算坐标系")
                landmarks, curvature_data = extract_tooth_landmarks_with_curvature(tooth_points, tooth_type, tooth_axes)
                print(f"识别到 {len(landmarks)} 个特征点")
                fig = plt.figure(figsize=(15, 5))
                fig.suptitle(f'Tooth #{tooth_num} ({tooth_type}) Analysis', fontsize=14)
                ax1 = fig.add_subplot(131, projection='3d')
                scatter = ax1.scatter(tooth_points[:, 0], tooth_points[:, 1], tooth_points[:, 2],
                                   c=curvature_data['gaussian_curvature'],
                                   cmap='coolwarm')
                plt.colorbar(scatter, ax=ax1)
                ax1.set_title('Gaussian Curvature')
                ax2 = fig.add_subplot(132, projection='3d')
                scatter = ax2.scatter(tooth_points[:, 0], tooth_points[:, 1], tooth_points[:, 2],
                                   c=curvature_data['mean_curvature'],
                                   cmap='coolwarm')
                plt.colorbar(scatter, ax=ax2)
                ax2.set_title('Mean Curvature')
                ax3 = fig.add_subplot(133, projection='3d')
                ax3.scatter(tooth_points[:, 0], tooth_points[:, 1], tooth_points[:, 2],
                          c='black', alpha=0.1, s=1)
                colors = {
                    'incisal_point': 'red',
                    'buccal_mesial_cusp': 'blue',
                    'buccal_distal_cusp': 'green',
                    'lingual_mesial_cusp': 'yellow',
                    'lingual_distal_cusp': 'purple',
                    'central_fossa': 'black'
                }
                for name, point in landmarks.items():
                    if name in colors:
                        ax3.scatter(point[0], point[1], point[2],
                                 c=colors[name], s=100, label=name)
                plot_coordinate_system(ax3, origin, axes)
                ax3.legend()
                ax3.set_title('Landmarks and Coordinate System')
                for ax in [ax1, ax2, ax3]:
                    ax.view_init(elev=20, azim=45)
                    ax.set_box_aspect([1,1,1])
                plt.tight_layout()
                plt.show()
                input("按回车键继续查看下一个牙齿...")
                plt.close()
            except Exception as e:
                print(f"处理文件 {stl_file} 时出错：{str(e)}")
                continue
    except Exception as e:
        print(f"程序执行出错：{str(e)}")

if __name__ == "__main__":
    stl_folder = "/devdata/dzx_data/DiskF/all_raw_data/FADA43/tooth/down"
    tooth_info = "/devdata/dzx_data/DiskF/all_raw_data/FADA43/toothInfo.txt"
    test_landmark_detection(stl_folder, tooth_info)