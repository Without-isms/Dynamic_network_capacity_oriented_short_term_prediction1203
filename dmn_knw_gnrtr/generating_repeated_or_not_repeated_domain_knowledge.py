import pickle
import os
import numpy as np
import torch
from metro_data_convertor.Find_project_root import Find_project_root

def generating_repeated_or_not_repeated_domain_knowledge(base_dir, od_type, prefix, repeated_or_not_repeated, seq_len):
    Time_DepartFreDic_Array_file_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_Time_DepartFreDic_Array.pkl')
    Date_and_time_OD_path_dic_file_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_Date_and_time_OD_path_dic.pkl')
    Time_DepartFreDic_Matrix_file_path=os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_Time_DepartFreDic_Matrix.pkl')
    Date_and_time_OD_path_Matrix_file_path=os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_Date_and_time_OD_path_Matrix.pkl')
    OD_feature_array_file_path=os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_OD_feature_array.pkl')
    with open(Time_DepartFreDic_Array_file_path, 'rb') as f:
        Time_DepartFreDic_Array = pickle.load(f, errors='ignore')

    if repeated_or_not_repeated=="not_repeated":
        with open(Date_and_time_OD_path_dic_file_path, 'rb') as f:
            Date_and_time_OD_path_dic = pickle.load(f, errors='ignore')

        Time_DepartFreDic_Matrix = []
        Date_and_time_OD_path_Matrix = []
        keys_sorted = sorted(Time_DepartFreDic_Array.keys())
        for idx in range(seq_len, len(keys_sorted)):
            temp_list = [Time_DepartFreDic_Array[keys_sorted[idx - j]] for j in range(seq_len)]
            Date_and_time_OD_path_temp_list = [Date_and_time_OD_path_dic[keys_sorted[idx - j]] for j in range(seq_len)]

            Time_DepartFreDic_Matrix.append(temp_list)
            Date_and_time_OD_path_Matrix.append(Date_and_time_OD_path_temp_list)

        output_file_Time_DepartFreDic_Matrix = open(Time_DepartFreDic_Matrix_file_path, 'wb')
        torch.save(Time_DepartFreDic_Matrix, output_file_Time_DepartFreDic_Matrix)

        output_Date_and_time_OD_path_Matrix = open(Date_and_time_OD_path_Matrix_file_path, 'wb')
        torch.save(Date_and_time_OD_path_Matrix, output_Date_and_time_OD_path_Matrix)
    else:
        sparse_tensors_2D = torch.load(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_tensors_OD_3_paths_all_OD.pt'))
        sparse_tensors_3D = torch.load(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_3d_tensor_list.pt'))
        sparse_tensors_4D = torch.load(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_4d_tensor_list.pt'))
        sparse_tensors_5D = torch.load(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_sparse_5d_tensor.pt'))

        #with open(os.path.join(base_dir, f'{prefix}_OD_path_compressed_array.pkl'), 'rb') as f:
            #OD_path_compressed_array = pickle.load(f, errors='ignore')

        with open(OD_feature_array_file_path, 'rb') as f:
            OD_feature_array = pickle.load(f, errors='ignore')

        Time_DepartFre_Array = Time_DepartFreDic_Array[next(iter(Time_DepartFreDic_Array))]

        log_dir = os.path.join(f"{base_dir}{os.path.sep}OD", prefix + '.pkl')
        with open(log_dir, 'rb') as f:
            matrix = pickle.load(f)['Time_DepartFreDic_Matrix']
        T = matrix.shape[0]
        #repeated_OD_path_compressed_array = np.repeat(np.expand_dims(OD_path_compressed_array, axis=0),
                                                      #matrix.shape[0] + seq_len, axis=0)
        repeated_OD_feature_array = np.repeat(np.expand_dims(OD_feature_array, axis=0), matrix.shape[0] + seq_len, axis=0)
        repeated_2D_OD_path_lst = [sparse_tensors_2D for _ in range(matrix.shape[0] + seq_len)]
        repeated_3D_OD_path_lst = [sparse_tensors_3D for _ in range(matrix.shape[0] + seq_len)]
        repeated_4D_OD_path_lst = [sparse_tensors_4D for _ in range(matrix.shape[0] + seq_len)]
        repeated_5D_OD_path_lst = [sparse_tensors_5D for _ in range(matrix.shape[0] + seq_len)]
        repeated_Time_DepartFre_Array = np.repeat(np.expand_dims(Time_DepartFre_Array, axis=0), matrix.shape[0] + seq_len,
                                                  axis=0)

        #repeated_OD_path_compressed_array_ = []
        repeated_OD_feature_array_ = []
        repeated_2D_OD_path_lst_ = []
        repeated_3D_OD_path_lst_ = []
        repeated_4D_OD_path_lst_ = []
        repeated_5D_OD_path_lst_ = []
        repeated_Time_DepartFre_Array_ = []
        for i in range(0, T):
            #temp_array = np.array([repeated_OD_path_compressed_array[i + j] for j in range(seq_len)])
            features_temp_array = np.array([repeated_OD_feature_array[i + j] for j in range(seq_len)])
            temp_tdf_array = np.array([repeated_Time_DepartFre_Array[i + j] for j in range(seq_len)])
            #repeated_OD_path_compressed_array_.append(temp_array)
            repeated_OD_feature_array_.append(features_temp_array)
            repeated_Time_DepartFre_Array_.append(temp_tdf_array)

            temp_2D_path_lst = [repeated_2D_OD_path_lst[i + j] for j in range(seq_len)]
            repeated_2D_OD_path_lst_.append(temp_2D_path_lst)

            temp_3D_path_lst = [repeated_3D_OD_path_lst[i + j] for j in range(seq_len)]
            repeated_3D_OD_path_lst_.append(temp_3D_path_lst)

            temp_4D_path_lst = [repeated_4D_OD_path_lst[i + j] for j in range(seq_len)]
            repeated_4D_OD_path_lst_.append(temp_4D_path_lst)

            temp_5D_path_lst = [repeated_5D_OD_path_lst[i + j] for j in range(seq_len)]
            repeated_5D_OD_path_lst_.append(temp_5D_path_lst)

        #repeated_OD_path_compressed_array_ = np.array(repeated_OD_path_compressed_array_, dtype=np.int16)
        repeated_OD_feature_array_ = np.array(repeated_OD_feature_array_)
        repeated_Time_DepartFre_Array_ = np.array(repeated_Time_DepartFre_Array_, dtype=np.float16)

        output_file_path_2D = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_sparse_2D_tensors.pt'), 'wb')
        torch.save(repeated_2D_OD_path_lst_, output_file_path_2D)

        output_file_path_3D = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_sparse_3D_tensors.pt'),'wb')
        torch.save(repeated_3D_OD_path_lst_, output_file_path_3D)

        output_file_path_4D = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_sparse_4D_tensors.pt'),'wb')
        torch.save(repeated_4D_OD_path_lst_, output_file_path_4D)

        output_file_path_5D = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_sparse_5D_tensors.pt'), 'wb')
        torch.save(repeated_5D_OD_path_lst_, output_file_path_5D)

        #file1 = open(os.path.join(base_dir, prefix + '_OD_path_compressed_array.pkl'), 'wb')
        #pickle.dump(repeated_OD_path_compressed_array_, file1)

        file2 = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_OD_feature_array.pkl'), 'wb')
        pickle.dump(repeated_OD_feature_array_, file2)

        file_tdf = open(os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_repeated_Time_DepartFre_Array.pkl'), 'wb')
        pickle.dump(repeated_Time_DepartFre_Array_, file_tdf)

"""project_root = Find_project_root()
base_dir = os.path.join(project_root, 'data', 'suzhou')
prefix = "train"
od_type="OD"
repeated_or_not_repeated= "not_repeated"
seq_len=4
generating_repeated_or_not_repeated_domain_knowledge(base_dir, od_type, prefix, repeated_or_not_repeated, seq_len)"""