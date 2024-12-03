import os
from metro_data_convertor.Find_project_root import Find_project_root
import shutil


project_root = Find_project_root()
base_path = os.path.join(project_root, "data", "suzhou", "DO")

if os.path.exists(base_path):
    if os.path.isdir(base_path):
        shutil.rmtree(base_path)
        print(f"{base_path} 目录及其内容已成功删除")
    else:
        os.remove(base_path)
        print(f"{base_path} 文件已成功删除")
else:
    print(f"{base_path} 不存在")


project_root = Find_project_root()
base_path = os.path.join(project_root, "data", "suzhou")

file_names = [
    r"OD\test_sparse_4d_tensor_list.pt",
    r"OD\test_sparse_5d_tensor.pt",
    r"OD\train_sparse_4d_tensor_list.pt",
    r"OD\train_sparse_5d_tensor.pt",
    r"OD\val_sparse_4d_tensor_list.pt",
    r"OD\val_sparse_5d_tensor.pt",
    r"OD\RecurrentGCN_model.pth"
]

# 遍历删除文件
for file_name in file_names:
    file_path = os.path.join(base_path, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} 已成功删除")
    else:
        print(f"{file_path} 不存在")

import tempfile

# 获取系统的临时目录路径
system_temp_dir = tempfile.gettempdir()

# 遍历系统临时目录中的所有文件和子目录
for root, dirs, files in os.walk(system_temp_dir):
    for file in files:
        # 打印每个文件的完整路径
        print(os.path.join(root, file))

# 获取系统的临时目录路径
system_temp_dir = tempfile.gettempdir()

# 遍历系统临时目录中的所有文件和子目录
for root, dirs, files in os.walk(system_temp_dir, topdown=False):
    # 删除文件
    for file in files:
        file_path = os.path.join(root, file)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # 删除空子目录
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        try:
            os.rmdir(dir_path)
            print(f"Deleted directory: {dir_path}")
        except OSError as e:
            print(f"Directory {dir_path} is not empty or can't be deleted: {e}")

import torch

# 创建一个 3D 稀疏矩阵，形状为 (1000, 1000, 1000)
indices = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
values = torch.tensor([1.0, 2.0, 3.0])
sparse_tensor = torch.sparse_coo_tensor(indices, values, (1000, 1000, 1000))

print(sparse_tensor)
print(sparse_tensor.shape)

import torch

indices = torch.tensor([
    [0, 1, 2],   # 第一维度的索引
    [0, 1, 2],   # 第二维度的索引
    [0, 1, 2],   # 第三维度的索引
    [0, 1, 2],   # 第四维度的索引
    [0, 1, 2]    # 第五维度的索引
])

values = torch.tensor([10.0, 20.0, 30.0])

sparse_tensor = torch.sparse_coo_tensor(indices, values, (100, 100, 100, 100, 100))

print(sparse_tensor)
print(sparse_tensor.shape)



import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.device_count())

