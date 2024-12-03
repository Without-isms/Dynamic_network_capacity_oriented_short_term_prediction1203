import numpy as np
import pickle

# 加载OD_path_dic
with open('OD_path_dic.pkl', 'rb') as f:
    OD_path_dic = pickle.load(f)

# 加载station_manager_dict_no_11
with open('station_manager_dict_no_11.pkl', 'rb') as f:
    station_manager_dict = pickle.load(f)

# 提取station_index字典
station_index = station_manager_dict['station_index']

# 获取站点数量
num_stations = len(station_index)

# 初始化OD_path_array、OD_feature_array和OD_path_compressed_array
OD_path_array = np.zeros((num_stations, num_stations, 3, num_stations, num_stations))
OD_feature_array = np.zeros((num_stations, num_stations, 3, 2))
OD_path_compressed_array = np.full((num_stations, num_stations, 3, 50), -1)

# 遍历OD_path_dic中的每一个键值对
for (origin, destination), list_of_paths in OD_path_dic.items():
    # 获取origin和destination的索引
    origin_idx = station_index[origin]
    destination_idx = station_index[destination]

    # 初始化3*站点数*站点数的路径矩阵列表和3*2的特征矩阵列表，以及3*50的压缩矩阵列表
    path_matrices = []
    feature_matrices = []
    compressed_matrices = []

    for path_dict in list_of_paths:
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((num_stations, num_stations))

        # 初始化压缩路径矩阵
        compressed_matrix = np.full(50, -1)

        # 获取站点访问序列
        station_visit_sequence = path_dict['station_visit_sequence']

        # 填充邻接矩阵和压缩路径矩阵
        for i in range(len(station_visit_sequence) - 1):
            current_station = station_visit_sequence[i]['index']
            next_station = station_visit_sequence[i + 1]['index']
            adjacency_matrix[current_station, next_station] = 1
            if i < 50:
                compressed_matrix[i] = current_station

        if len(station_visit_sequence) <= 50:
            # Add last station if it fits within the 50 slots
            compressed_matrix[len(station_visit_sequence) - 1] = station_visit_sequence[-1]['index']

        # 将生成的邻接矩阵添加到路径矩阵列表中
        path_matrices.append(adjacency_matrix)

        # 初始化特征矩阵 (站点数, 换乘数)
        feature_matrix = np.zeros(2)

        # 填充特征矩阵
        feature_matrix[0] = path_dict['number_of_stations']
        feature_matrix[1] = path_dict['number_of_transfers']

        # 将生成的特征矩阵添加到特征矩阵列表中
        feature_matrices.append(feature_matrix)

        # 将生成的压缩矩阵添加到压缩矩阵列表中
        compressed_matrices.append(compressed_matrix)

    # 如果路径数量不足三条，重复第一条路径直到数量为三
    while len(path_matrices) < 3:
        path_matrices.append(path_matrices[0])
        feature_matrices.append(feature_matrices[0])
        compressed_matrices.append(compressed_matrices[0])

    # 将3个邻接矩阵存入OD_path_array
    OD_path_array[origin_idx, destination_idx] = np.array(path_matrices)

    # 将3个特征矩阵存入OD_feature_array
    OD_feature_array[origin_idx, destination_idx] = np.array(feature_matrices)

    # 将3个压缩矩阵存入OD_path_compressed_array
    OD_path_compressed_array[origin_idx, destination_idx] = np.array(compressed_matrices)

# Trim OD_path_compressed_array by removing trailing -1 values
def trim_compressed_array(array):
    # Find the last index where not all values are -1
    last_valid_index = array.shape[-1] - 1
    for i in range(array.shape[-1] - 1, -1, -1):
        if not np.all(array[..., i] == -1):
            last_valid_index = i
            break
    # Slice the array to keep only up to the last valid index
    print("last_valid_index"+str(last_valid_index))
    return array[..., :last_valid_index + 1]

# Apply trimming to OD_path_compressed_array
OD_path_compressed_array = trim_compressed_array(OD_path_compressed_array)


# 检查结果
print("OD_path_array shape:", OD_path_array.shape)  # 应该是 (num_stations, num_stations, 3, num_stations, num_stations)
print("OD_feature_array shape:", OD_feature_array.shape)  # 应该是 (num_stations, num_stations, 3, 2)
print("OD_path_compressed_array shape:", OD_path_compressed_array.shape)  # 应该是 (num_stations, num_stations, 3, 50)

# 保存数组到pickle文件
with open('OD_path_array.pkl', 'wb') as f:
    pickle.dump(OD_path_array, f)

with open('OD_feature_array.pkl', 'wb') as f:
    pickle.dump(OD_feature_array, f)

with open('OD_path_compressed_array.pkl', 'wb') as f:
    pickle.dump(OD_path_compressed_array, f)

print("所有数组已保存。")
