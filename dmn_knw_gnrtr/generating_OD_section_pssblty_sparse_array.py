import numpy as np
import pickle
import torch


def compare_sparse_tensors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        print("The shapes of the tensors are inconsistent.")
        return False

    if not tensor1.is_sparse or not tensor2.is_sparse:
        print("At least one tensor is not a sparse tensor.")
        return False

    indices1 = tensor1._indices()
    indices2 = tensor2._indices()
    if not torch.equal(indices1, indices2):
        print("Indices are inconsistent.")
        return False

    values1 = tensor1.values()
    values2 = tensor2.values()
    if not torch.equal(values1, values2):
        print("Values of non-zero elements are inconsistent.")
        return False

    print("The two sparse tensors are consistent.")
    return True

def convert_to_sparse_tensor_lists(sparse_tensors_OD_3_paths_all_OD, num_stations, tensor_dim):
    sparse_tensor_list = []

    all_indices = []
    all_values = []

    for origin_idx in range(num_stations):
        for destination_idx in range(num_stations):
            for k in range(3):
                sparse_list = sparse_tensors_OD_3_paths_all_OD[origin_idx][destination_idx][k]
                for sparse_tensor in sparse_list:
                    sparse_tensor = sparse_tensor.coalesce()
                    indices = sparse_tensor.indices()
                    values = sparse_tensor.values()

                    for i in range(indices.size(1)):
                        if tensor_dim == 5:
                            all_indices.append(
                                [origin_idx, destination_idx, k, indices[0, i].item(), indices[1, i].item()])
                        elif tensor_dim == 4:
                            all_indices.append([destination_idx, k, indices[0, i].item(), indices[1, i].item()])
                        elif tensor_dim == 3:
                            all_indices.append([k, indices[0, i].item(), indices[1, i].item()])
                        all_values.append(values[i].item())

    if all_indices:
        all_indices = torch.tensor(all_indices).T
        all_values = torch.tensor(all_values)

        if tensor_dim == 5:
            sparse_shape = (num_stations, num_stations, 3, num_stations, num_stations)
        elif tensor_dim == 4:
            sparse_shape = (num_stations, 3, num_stations, num_stations)
        elif tensor_dim == 3:
            sparse_shape = (3, num_stations, num_stations)

        sparse_tensor = torch.sparse_coo_tensor(all_indices, all_values, sparse_shape).coalesce()
        sparse_tensor_list.append(sparse_tensor)

    return sparse_tensor_list

def generating_OD_section_pssblty_sparse_array(base_dir, prefix, od_type, station_manager_dict_name,
                                               Time_DepartFreDic_file_path, OD_path_dic_file_path,
                                               station_manager_dict_file_path, OD_feature_array_file_path,
                                               Date_and_time_OD_path_dic_file_path):
    with open(Time_DepartFreDic_file_path, 'rb') as file:
        Time_DepartFreDic = pickle.load(file)

    with open(OD_path_dic_file_path, 'rb') as f:
        OD_path_dic = pickle.load(f)

    with open(station_manager_dict_file_path, 'rb') as f:
        station_manager_dict = pickle.load(f)

    station_index = station_manager_dict['station_index']
    num_stations = len(station_index)

    index_station = station_manager_dict['index_station']

    Date_and_time_OD_path_dic={}
    OD_feature_array_dic={}

    for Time_DepartFre_key, Time_DepartFre_value in Time_DepartFreDic.items():
        date_and_time = Time_DepartFre_key
        OD_feature_array = np.zeros((num_stations, num_stations, 3, 2))
        # OD_path_array = np.zeros((num_stations, num_stations, 3, num_stations, num_stations), dtype=np.int8)
        # OD_path_compressed_array = np.full((num_stations, num_stations, 3, 50), -1)
        sparse_tensors_OD_3_paths_all_OD = [[[[] for _ in range(3)] for _ in range(num_stations)] for _ in
                                            range(num_stations)]
        if Time_DepartFre_value == {}:
            sparse_shape = (num_stations, num_stations, 3, num_stations, num_stations)
            sparse_tensor = torch.sparse_coo_tensor(indices=torch.empty((5, 0), dtype=torch.int32),
                                                    values=torch.empty(0),
                                                    size=sparse_shape)
            Date_and_time_OD_path_dic[date_and_time] = sparse_tensor
            OD_feature_array_dic[date_and_time] = OD_feature_array
        else:
            for (origin, destination), list_of_paths in OD_path_dic.items():
                origin_idx = station_index[origin]
                destination_idx = station_index[destination]
                path_matrices = []
                feature_matrices = []
                compressed_matrices = []

                for path_dict in list_of_paths:
                    adjacency_matrix = np.zeros((num_stations, num_stations), dtype=np.int8)
                    compressed_matrix = np.full(50, -1)
                    feature_matrix = np.zeros(2)
                    station_visit_sequence = path_dict['station_visit_sequence']
                    for i in range(len(station_visit_sequence) - 1):
                        current_station = station_visit_sequence[i]['index']
                        next_station = station_visit_sequence[i + 1]['index']
                        if (index_station[current_station],index_station[next_station]) not in Time_DepartFre_value:
                            break
                        else:
                            adjacency_matrix[current_station, next_station] = 1
                            if i < 50:
                                compressed_matrix[i] = current_station
                    if len(station_visit_sequence) <= 50:
                        compressed_matrix[len(station_visit_sequence) - 1] = station_visit_sequence[-1]['index']
                    path_matrices.append(adjacency_matrix)
                    compressed_matrices.append(compressed_matrix)
                    feature_matrix[0] = path_dict['number_of_stations']
                    feature_matrix[1] = path_dict['number_of_transfers']
                    feature_matrices.append(feature_matrix)

                if path_matrices==[]:
                    adjacency_matrix = np.zeros((num_stations, num_stations), dtype=np.int8)
                    path_matrices.append(adjacency_matrix)
                    feature_matrix = np.zeros(2)
                    feature_matrices.append(feature_matrix)
                    compressed_matrix = np.full(50, -1)
                    compressed_matrices.append(compressed_matrix)

                while len(path_matrices) < 3:
                    path_matrices.append(path_matrices[0])
                    feature_matrices.append(feature_matrices[0])
                    compressed_matrices.append(compressed_matrices[0])

                # OD_path_array[origin_idx, destination_idx] = np.array(path_matrices)
                OD_feature_array[origin_idx, destination_idx] = np.array(feature_matrices)
                # OD_path_compressed_array[origin_idx, destination_idx] = np.array(compressed_matrices)

                for k in range(3):
                    adj_matrix = path_matrices[k]
                    indices = np.nonzero(adj_matrix)
                    values = adj_matrix[indices]
                    indices = np.array(indices)
                    indices = torch.tensor(indices, dtype=torch.long)
                    values = torch.tensor(values, dtype=torch.int8)
                    sparse_tensor = torch.sparse_coo_tensor(indices, values, (num_stations, num_stations)).coalesce()
                    sparse_tensors_OD_3_paths_all_OD[origin_idx][destination_idx][k].append(sparse_tensor)

            def trim_compressed_array(array):
                last_valid_index = array.shape[-1] - 1
                for i in range(array.shape[-1] - 1, -1, -1):
                    if not np.all(array[..., i] == -1):
                        last_valid_index = i
                        break
                print("last_valid_index" + str(last_valid_index))
                return array[..., :last_valid_index + 1]

            # OD_path_compressed_array = trim_compressed_array(OD_path_compressed_array)

            # with open('OD_path_array.pkl', 'wb') as f:
            # pickle.dump(OD_path_array, f)

            with open(OD_feature_array_file_path, 'wb') as f:
                pickle.dump(OD_feature_array, f)

            sparse_3d_tensor_list = convert_to_sparse_tensor_lists(sparse_tensors_OD_3_paths_all_OD, num_stations,
                                                                   tensor_dim=3)
            sparse_4d_tensor_list = convert_to_sparse_tensor_lists(sparse_tensors_OD_3_paths_all_OD, num_stations,
                                                                   tensor_dim=4)
            sparse_5d_tensor = convert_to_sparse_tensor_lists(sparse_tensors_OD_3_paths_all_OD, num_stations,
                                                              tensor_dim=5)

            OD_feature_array_dic[date_and_time] = OD_feature_array
            Date_and_time_OD_path_dic[date_and_time] = sparse_5d_tensor

            #with open('OD_path_compressed_array.pkl', 'wb') as f:
                #pickle.dump(OD_path_compressed_array, f)

            """torch.save(sparse_tensors_OD_3_paths_all_OD, os.path.join(base_dir,
                                                                      f'{od_type.upper()}{os.path.sep}{prefix}_sparse_tensors_OD_3_paths_all_OD.pt'))
            torch.save(sparse_5d_tensor[0],
                       os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_5d_tensor.pt'))
            torch.save(sparse_4d_tensor_list[0],
                       os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_4d_tensor_list.pt'))
            torch.save(sparse_3d_tensor_list[0],
                       os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_3d_tensor_list.pt'))"""


    with open(OD_feature_array_file_path, 'wb') as f:
        pickle.dump(OD_feature_array_dic, f)
    with open(Date_and_time_OD_path_dic_file_path, 'wb') as f:
        pickle.dump(Date_and_time_OD_path_dic, f)

'''project_root = Find_project_root()
base_dir = os.path.join(project_root, 'data', 'suzhou')
prefix = "train"
od_type="OD"
station_manager_dict_name='station_manager_dict_no_11.pkl'
graph_sz_conn_no_name='graph_sz_conn_no_11.pkl'
Time_DepartFreDic_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
OD_path_dic_file_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}OD_path_dic.pkl')
station_manager_dict_file_path = os.path.join(base_dir, f"{station_manager_dict_name}")
OD_feature_array_file_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_OD_feature_array_dic.pkl')
Date_and_time_OD_path_dic_file_path = os.path.join(base_dir,
                                                   f'{od_type.upper()}{os.path.sep}{prefix}_Date_and_time_OD_path_dic.pkl')
generating_OD_section_pssblty_sparse_array(base_dir, prefix, od_type, station_manager_dict_name,
                                               Time_DepartFreDic_file_path, OD_path_dic_file_path,
                                               station_manager_dict_file_path, OD_feature_array_file_path,
                                               Date_and_time_OD_path_dic_file_path)'''