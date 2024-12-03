# 导入模块
import pymysql
import numpy as np
import pandas as pd
import datetime
import pickle
import os
from collections import defaultdict
import pickle
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import os
import numpy as np
import pickle
import sys


def generate_OD_DO_array(df, x_y_time, sz_conn_to_station_index, station_index_to_sz_conn, 
                         base_dir, prefix, od_or_do, station_manager_dict_name):
    """
    Generate OD or DO arrays.
    :param df: DataFrame
    :param x_y_time: List of timestamps
    :param sz_conn_to_station_index: Starting station index dictionary
    :param station_index_to_sz_conn: Dictionary mapping station index to station
    :param base_dir: Base directory
    :param prefix: File prefix
    :param od_or_do: 'od' for generating OD matrix, 'do' for generating DO matrix
    :return: Generated dictionary
    """

    output_dir = os.path.join(base_dir, od_or_do.upper())
    os.makedirs(output_dir, exist_ok=True)
    station_manager_dict_file_path = f"data{os.path.sep}suzhou{os.path.sep}{station_manager_dict_name}"
    result_array_file_path = os.path.join(output_dir, f'{prefix}_result_array_{od_or_do.upper()}.pkl')

    with open(station_manager_dict_file_path, 'rb') as f:
        station_manager_dict = pickle.load(f)

    T_N_D_ODs = {}
    Trip_Production_In_Station_or_Out_Station = {}
    Trip_Attraction_In_Station_or_Out_Station = {}

    for time in tqdm(x_y_time, desc=f"Processing Time Stamps for {od_or_do.upper()}"):

        df_time = df[df['in_count_time'] == time]

        N = len(sz_conn_to_station_index)
        od_matrix = np.zeros((N, N))
        trip_Production_matrix = np.zeros((N, 1))
        trip_attraction_matrix = np.zeros((N, 1))

        for i, start_or_end_station_list in sz_conn_to_station_index.items():
            if start_or_end_station_list[0] == 0:
                print("there is no station named" + str(station_index_to_sz_conn[start_or_end_station_list[0]]))
                combined_string = ''.join(station_manager_dict['index_station'][element] for element in station_index_to_sz_conn[start_or_end_station_list[0]])
                print("there is no station named" + combined_string)
                continue

            same_start_station_dif_index_dest_counts = pd.Series()
            same_start_station_dif_index_dest_counts_all = 0
            same_end_station_dif_index_start_counts_all = 0
            for start_or_end_station in start_or_end_station_list:
                if od_or_do == "od":
                    df_start_or_end = df_time[df_time['in_station_id'] == start_or_end_station]
                    dest_counts = df_start_or_end.groupby('out_station_id')['count_od'].sum()
                    dest_counts_all = df_start_or_end['count_od'].sum()
                    df_terminal = df_time[df_time['out_station_id'] == start_or_end_station]
                    orig_counts = df_terminal.groupby('in_station_id')['count_od'].sum()
                    orig_counts_all = df_start_or_end['count_od'].sum()
                else:
                    df_start_or_end = df_time[df_time['out_station_id'] == start_or_end_station]
                    dest_counts = df_start_or_end.groupby('in_station_id')['count_od'].sum()
                    dest_counts_all = df_start_or_end['count_od'].sum()
                    df_terminal = df_time[df_time['in_station_id'] == start_or_end_station]
                    orig_counts = df_start_or_end.groupby('out_station_id')['count_od'].sum()
                    orig_counts_all = df_start_or_end['count_od'].sum()

                if dest_counts.sum() == 0:
                    continue

                idx = same_start_station_dif_index_dest_counts.index.union(dest_counts.index)
                same_start_station_dif_index_dest_counts = same_start_station_dif_index_dest_counts.reindex(idx).fillna(0)
                dest_counts = dest_counts.reindex(idx).fillna(0)
                same_start_station_dif_index_dest_counts += dest_counts
                same_start_station_dif_index_dest_counts_all += dest_counts_all
                same_end_station_dif_index_start_counts_all += orig_counts_all

            for same_start_station_dest_counts, count in same_start_station_dif_index_dest_counts.items():
                out_station_index = station_index_to_sz_conn[same_start_station_dest_counts]
                od_matrix[i, out_station_index] = count

            trip_Production_matrix[i] = same_start_station_dif_index_dest_counts_all
            trip_attraction_matrix[i] = same_end_station_dif_index_start_counts_all

        T_N_D_ODs[time] = od_matrix
        Trip_Production_In_Station_or_Out_Station[time] = trip_Production_matrix
        Trip_Attraction_In_Station_or_Out_Station[time] = trip_attraction_matrix
    intermediate_dic = {
        'T_N_D_ODs': T_N_D_ODs,
        'Trip_Production_In_Station_or_Out_Station':Trip_Production_In_Station_or_Out_Station,
        'Trip_Attraction_In_Station_or_Out_Station': Trip_Attraction_In_Station_or_Out_Station
    }
    # In fact, only T_N_D_ODs is needed here, but it's included for faster execution.

    with open(result_array_file_path, 'wb') as f:
        pickle.dump(intermediate_dic, f)

    return intermediate_dic

def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"{filename} has been created successfully.")

def print_array_info(array):
    print(array.shape)

def generate_T_N_D_ODs(result_array_all, D):
    T, N, _ = result_array_all.shape
    result_array = np.zeros((T, N, D + 1), dtype=result_array_all.dtype)
    for in_station in range(N):
        out_sums = np.sum(result_array_all[:, in_station, :], axis=0)
        top_D_indices = np.argsort(out_sums)[::-1][:D]
        remaining_indices = np.argsort(out_sums)[::-1][D:]
        for t in range(T):
            top_D_values = result_array_all[t, in_station, top_D_indices]
            remaining_values = result_array_all[t, in_station, remaining_indices]
            result_array[t, in_station, :D] = top_D_values
            result_array[t, in_station, D] = np.sum(remaining_values)

    return result_array

def process_data(D, base_dir, prefix, od_type, str_prdc_attr, seq_len):
    dir_path = os.path.join(base_dir, od_type.upper())
    os.makedirs(dir_path, exist_ok=True)
    result_array_file_path = os.path.join(dir_path, f'{prefix}_result_array_{od_type.upper()}.pkl')
    signal_dict_file_path = f'{prefix}_{od_type.upper()}_{str_prdc_attr}_signal_dict.pkl'
    signal_dict_array_file_path = os.path.join(dir_path, f'{prefix}_{od_type.upper()}_signal_dict_array.pkl')
    train_history_short_filename = os.path.join(dir_path, f'{prefix}_history_short.pkl')
    train_history_long_filename = os.path.join(dir_path, f'{prefix}_history_long.pkl')
    if od_type=="od":
        train_dict_file_path = os.path.join(dir_path, f'{prefix}.pkl')
    else:
        train_dict_file_path = os.path.join(dir_path, f'{prefix}_do.pkl')

    with open(result_array_file_path, 'rb') as f:
        intermediate_dic = pickle.load(f, errors='ignore')
        T_N_D_ODs = intermediate_dic['T_N_D_ODs']
        sorted_times = sorted(T_N_D_ODs.keys())
        result_array_all = np.array([T_N_D_ODs[time] for time in sorted_times])
        result_array = generate_T_N_D_ODs(result_array_all, D)

        with open(os.path.join(dir_path, signal_dict_file_path), 'rb') as f:
            signal_dict = pickle.load(f, errors='ignore')
            features = signal_dict["features"]
            additional_feature = signal_dict["additional_feature"]


    T = len(sorted_times)
    finished = []
    unfinished = []
    xtime = []
    features_ = []
    additional_feature_ =[]
    for i in range(seq_len, T):
        temp_array = np.array([result_array[i - j] for j in range(seq_len)])
        features_temp_array = np.array([features[i - j] for j in range(seq_len)])
        additional_feature_temp_array = np.array([additional_feature[i - j] for j in range(seq_len)])
        finished.append(temp_array)
        features_.append(features_temp_array)
        additional_feature_.append(additional_feature_temp_array)

        unfinished_temp_array_list = []

        """
        for j in range(seq_len):
            current_array = result_array[i - j]
            print(f"result_array[{i - j}] shape: {current_array.shape}")

            summed_array = current_array.sum(axis=1, keepdims=True)
            print(f"summed_array shape: {summed_array.shape}")

            unfinished_temp_array_list.append(summed_array)

        unfinished_temp_array = np.array(unfinished_temp_array_list)
        """
        unfinished_temp_array = np.array([result_array[i - j].sum(axis=1, keepdims=True) for j in range(seq_len)])
        unfinished.append(unfinished_temp_array)
        xtime.append(sorted_times[i])

    finished_array = np.array(finished[:-1])
    unfinished_array = np.array(unfinished[:-1])
    xtime = np.array(xtime[:-1])
    features_ = np.array(features_[:-1])
    additional_feature_ = np.array(additional_feature_[:-1])

    signal_dict_array = {
        'features': features_,
        'additional_feature': additional_feature_
    }

    with open(signal_dict_array_file_path, 'wb') as f:
        pickle.dump(signal_dict_array, f)

    y = []
    ytime = []
    for i in range(1, T - seq_len):
        temp_array = np.array([result_array[i + j] for j in range(seq_len)])
        y.append(temp_array)
        ytime.append(sorted_times[i + seq_len])

    y_array = np.array(y)
    ytime = np.array(ytime)

    train_dict = {
        'finished': finished_array,
        'unfinished': unfinished_array,
        'y': y_array,
        'xtime': xtime,
        'ytime': ytime
    }

    print_array_info(finished_array)
    print_array_info(unfinished_array)
    print_array_info(y_array)
    print_array_info(xtime)
    print_array_info(ytime)
    save_to_pickle(train_dict, train_dict_file_path)

    if od_type == "od":
        xtime = np.array([np.datetime64(ts) for ts in xtime])
        one_week_ns = np.timedelta64(1, 'W')
        one_day_ns = np.timedelta64(1, 'D')

        train_history_short = np.zeros_like(finished_array)
        train_history_long = np.zeros_like(finished_array)

        for i, current_time in enumerate(xtime):
            target_time_week = current_time - one_week_ns
            target_time_day = current_time - one_day_ns

            closest_index_week = np.abs(xtime - target_time_week).argmin()
            if xtime[closest_index_week] > current_time:
                while closest_index_week > 0 and xtime[closest_index_week] > current_time:
                    closest_index_week -= 1

            closest_index_day = np.abs(xtime - target_time_day).argmin()
            if xtime[closest_index_day] > current_time:
                while closest_index_day > 0 and xtime[closest_index_day] > current_time:
                    closest_index_day -= 1

            train_history_short[i] = finished_array[closest_index_week]
            train_history_long[i] = finished_array[closest_index_day]

        train_history_short_dict = {'history': train_history_short}

        print_array_info(train_history_short)
        save_to_pickle(train_history_short_dict, train_history_short_filename)

        train_history_long_dict = {'history': train_history_long}
        print_array_info(train_history_long)
        save_to_pickle(train_history_long_dict, train_history_long_filename)

def Connect_to_SQL(prefix, train_sql, test_sql, val_sql, station_manager_dict_name,
                   host, user, password, database):
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(f'{parent_dir}{os.path.sep}metro_components')

    with open(os.path.join(f"data{os.path.sep}suzhou{os.path.sep}{station_manager_dict_name}"), 'rb') as f:
        station_manager_dict = pickle.load(f)
        station_index_sz = station_manager_dict['station_index']

    """conn = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="305qwe!@#$%^&*",
        database="suzhoudata0513",
        charset="utf8")"""
    import mysql.connector
    from mysql.connector import Error
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
    )

    print(conn)
    print(type(conn))
    cursor = conn.cursor()
    if(prefix=="train"):
        # sql = "SELECT * FROM suzhou20230512_16_all"  # train
        sql = train_sql
    elif(prefix=="test"):
        sql = test_sql
    elif(prefix=="val"):
        sql = val_sql
    else:
        sql = "SELECT * FROM suzhou202304_all"
    # sql = "SELECT * FROM suzhou202303"
    cursor.execute(sql)

    results = cursor.fetchall()

    cursor.close()
    conn.close()

    columns = ["date", "in_station_id", "in_station_name", "out_station_id", "out_station_name", "in_count_time",
               "count_od"]
    df = pd.DataFrame(results, columns=columns)

    in_station_dict = defaultdict(set)
    out_station_dict = defaultdict(set)

    for idx, row in df.iterrows():
        in_station_dict[row["in_station_name"]].add(row["in_station_id"])
        out_station_dict[row["out_station_name"]].add(row["out_station_id"])

    in_station_dict = {k: list(v) for k, v in in_station_dict.items()}
    out_station_dict = {k: list(v) for k, v in out_station_dict.items()}

    in_station_dict = dict(in_station_dict)
    out_station_dict = dict(out_station_dict)

    in_station_name = df["in_station_name"].unique()
    out_station_name = df["out_station_name"].unique()

    sz_conn_to_station_index = {}
    for key, value in station_index_sz.items():
        if key in out_station_dict:
            sz_conn_to_station_index[value] = out_station_dict[key]
        elif key in in_station_dict:
            sz_conn_to_station_index[value] = in_station_dict[key]
        else:
            sz_conn_to_station_index[value] = [0]
        station_mapping = {
            "新市桥": [521],
            "南门": [522],
            "南园北路": [523],
            "竹辉桥": [524],
            "荷花荡": [525],
            "黄天荡": [526],
            "金厍桥": [527],
            "星波街": [528],
            "李公堤南": [529],
            "金湖": [530],
            "华莲": [531],
            "斜塘": [532],
            "苏州奥体中心": [533],
            "方洲公园": [534],
            "星塘街": [535],
            "龙墩": [536],
            "东沙湖": [537],
            "葑亭大道": [538],
            "阳澄湖南": [539],
            "双桥": [0],
            "红庄": [0]
            # ,"津桥": [0]
        }
        if key in station_mapping:
            if sz_conn_to_station_index[value] == [0]:
                sz_conn_to_station_index[value] = station_mapping[key]
            else:
                sz_conn_to_station_index[value].extend(station_mapping[key])

    station_index_to_sz_conn = {}
    for key, value_list in sz_conn_to_station_index.items():
        if isinstance(value_list, list):
            for value in value_list:
                if value in station_index_to_sz_conn:
                    station_index_to_sz_conn[value].append(key)
                else:
                    station_index_to_sz_conn[value] = [key]
        else:
            if value_list in station_index_to_sz_conn:
                station_index_to_sz_conn[value_list].append(key)
            else:
                station_index_to_sz_conn[value_list] = [key]

    df["in_count_time"] = pd.to_datetime(df["in_count_time"]).astype('datetime64[ns]')

    x_y_time = df["in_count_time"].unique()
    x_y_time = pd.DatetimeIndex(x_y_time).sort_values()
    return df,x_y_time,sz_conn_to_station_index,station_index_to_sz_conn

"""
D = 26 
base_dir = "suzhou"
prefix = "train"
train_sql="SELECT * FROM suzhou2023_0509"
test_sql="SELECT * FROM suzhou20230511_all"
val_sql="SELECT * FROM suzhou20230510_all"
df,x_y_time,sz_conn_to_station_index,station_index_to_sz_conn=Connect_to_SQL(prefix, train_sql, test_sql, val_sql)
result_array_OD = generate_OD_DO_array(df, x_y_time, sz_conn_to_station_index, station_index_to_sz_conn, base_dir, prefix, 'od')
result_array_DO = generate_OD_DO_array(df, x_y_time, sz_conn_to_station_index, station_index_to_sz_conn, base_dir, prefix, 'do')
from PYGT_signal_Production_one_hot import PYGT_signal_Production
PYGT_signal_Production("OD", base_dir, prefix)
PYGT_signal_Production("DO", base_dir, prefix)
process_data(D, base_dir, prefix, "od", str_prdc_attr, seq_len)
process_data(D, base_dir, prefix, "do", str_prdc_attr, seq_len)
"""