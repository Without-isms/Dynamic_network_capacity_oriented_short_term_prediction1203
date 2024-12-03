import pickle
import os
import numpy as np
from metro_data_convertor.Find_project_root import Find_project_root

project_root = Find_project_root()
file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic_Array.pkl')
with open(file_path, 'rb') as f:
    Time_DepartFreDic_Array = pickle.load(f, errors='ignore')
    Time_DepartFre_Array = Time_DepartFreDic_Array[next(iter(Time_DepartFreDic_Array))]

base_dir = f"{project_root}{os.path.sep}data{os.path.sep}suzhou{os.path.sep}OD{os.path.sep}"
for str in ('train', 'val', 'test'):
    log_dir = os.path.join(base_dir, str + '.pkl')
    with open(log_dir, 'rb') as f:
        matrix = pickle.load(f)['finished']
    repeated_Time_DepartFre_Array = np.repeat(np.expand_dims(Time_DepartFre_Array, axis=0), matrix.shape[0], axis=0)

    T = matrix.shape[0]
    n = 4

    repeated_Time_DepartFre_Array_ = []
    for i in range(0, T):
        temp_array = np.array([repeated_Time_DepartFre_Array[i - j] for j in range(n)])
        repeated_Time_DepartFre_Array_.append(temp_array)

    repeated_Time_DepartFre_Array_ = np.array(repeated_Time_DepartFre_Array_, dtype=np.int16)
    repeated_Time_DepartFre_Array_ = np.array(repeated_Time_DepartFre_Array_)

    file1 = open(os.path.join(base_dir, str + '_repeated_Time_DepartFre_Array.pkl'), 'wb')
    pickle.dump(repeated_Time_DepartFre_Array_, file1)


