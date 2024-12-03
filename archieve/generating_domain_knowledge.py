from metro_data_convertor.Find_project_root import Find_project_root
from datetime import timedelta
from metro_data_convertor.Generating_logit_probabilities import Generating_logit_probabilities
from metro_data_convertor.Get_Time_DepartFreDic import Get_Time_DepartFreDic
from metro_data_convertor.Process_Time_DepartFreDic import Process_Time_DepartFreDic
from dmn_knw_gnrtr.fit_trip_generation_model import fit_trip_generation_model
import pickle
import numpy as np
import yaml
import os
import argparse
from train_button import read_cfg_file


def write_cfg_file(filename, cfg):
    with open(filename, 'w') as ymlfile:
        yaml.dump(cfg, ymlfile, default_flow_style=False, allow_unicode=True)

args = argparse.Namespace(config_filename=f'data{os.path.sep}config{os.path.sep}train_sz_dim26_units96_h4c512.yaml')
cfg = read_cfg_file(args.config_filename)


project_root = Find_project_root()
station_manager_dict_name='station_manager_dict_no_11.pkl'
graph_sz_conn_no_name='graph_sz_conn_no_11.pkl'
start_date_str = cfg['domain_knowledge']['start_date_str']
end_date_str = cfg['domain_knowledge']['end_date_str']
train_filename = os.path.join(project_root, 'data', 'suzhou', 'train_dict.pkl')
train_dict_file_path = os.path.join(project_root, 'data', 'suzhou', 'train_dict.pkl')
OD_path_visit_prob_dic_file_path = os.path.join(project_root, 'data', 'suzhou',
                                                'OD_path_visit_prob_dic.pkl')
Time_DepartFreDic_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
Time_DepartFreDic_Array_file_path = os.path.join(project_root, 'data', 'suzhou',
                                                             'Time_DepartFreDic_Array.pkl')
Time_DepartFreDic_filename = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
suzhou_sub_data_file_path = os.path.join(project_root, 'data', 'suzhou', 'suzhou_sub_data.xlsx')
excel_path = os.path.join(project_root, 'data', 'suzhou', 'Suzhou_zhandian_no_11.xlsx')
graph_sz_conn_root = os.path.join(project_root, 'data', 'suzhou', f'{graph_sz_conn_no_name}')
station_index_root = os.path.join(project_root, 'data', 'suzhou', 'station_index_no_11.pkl')
station_manager_dict_root = os.path.join(project_root, 'data', 'suzhou', 'station_manager_dict_no_11.pkl')
result_API_root = os.path.join(project_root, 'data', 'suzhou', 'result_API_modified.xlsx')
time_interval = timedelta(minutes=int(cfg['domain_knowledge']['timedelta_minutes']))
OD_path_visit_prob_array_file_path = os.path.join(project_root, 'data', 'suzhou', 'OD_path_visit_prob_array.pkl')
station_manager_dict_file_path = f"data{os.path.sep}suzhou{os.path.sep}{station_manager_dict_name}"


#用于生成所有OD pair所对应的所有path的信息，如果不更改路网，那么不用运行
"""Generating_Metro_Related_data(excel_path, graph_sz_conn_root, station_index_root, station_manager_dict_root,result_API_root,
                                  suzhou_sub_data_file_path, Time_DepartFreDic_filename, time_interval,
                              Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path, OD_path_visit_prob_array_file_path,
                              train_dict_file_path, OD_path_visit_prob_dic_file_path, train_filename, start_date_str, end_date_str)"""

#用于生成一个在确定的参数设置之下的logit分流结果，格式：(154,154,3,1)
Generating_logit_probabilities(train_dict_file_path, OD_path_visit_prob_dic_file_path,
                                           station_manager_dict_file_path, graph_sz_conn_root, station_manager_dict_root, station_index_root, result_API_root)
#获得一个以时间节点为key，以地铁发车频率相关信息为value的字典
Get_Time_DepartFreDic(suzhou_sub_data_file_path, Time_DepartFreDic_filename, time_interval,
                      excel_path, graph_sz_conn_root, station_index_root, start_date_str, end_date_str, station_manager_dict_root, result_API_root)

for prefix in ("train","test","val"):
    Time_DepartFreDic_Array_file_path = os.path.join(project_root, 'data', 'suzhou',
                                                     f'{prefix}_Time_DepartFreDic_Array.pkl')
    Process_Time_DepartFreDic(Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path, prefix)

# 结合Time_DepartFreDic，获取带有时变性的OD_visiting_prob矩阵
# 现有版本中弃用
# Reprocessing_OD_visiting_prob(OD_path_visit_prob_dic_file_path, OD_path_visit_prob_array_file_path)

base_dir = os.path.join(project_root, f"data{os.path.sep}suzhou")
od_type="OD"
train_sql= cfg['domain_knowledge']['train_sql']
test_sql= cfg['domain_knowledge']['test_sql']
val_sql= cfg['domain_knowledge']['val_sql']
repeated_or_not_repeated="not_repeated"
host='localhost'
user='root'
password='zxczxc@1234'
database='suzhoudata0513'
#RGCN是独立训练的
Using_lat_lng_or_index = cfg['model']['Using_lat_lng_or_index']
if Using_lat_lng_or_index=="lat_lng":
    RGCN_node_features = cfg['model']['num_nodes'] + 2
else:
    RGCN_node_features = cfg['model']['num_nodes'] + 1
RGCN_hidden_units =  int(cfg['domain_knowledge']['RGCN_hidden_units'])
RGCN_output_dim = int(cfg['domain_knowledge']['RGCN_output_dim'])
RGCN_K = int(cfg['domain_knowledge']['RGCN_K'])
lr = int(cfg['domain_knowledge']['lr'])
epoch_num = int(cfg['domain_knowledge']['epoch_num'])
seq_len = cfg['model']['seq_len']
train_ratio=float(cfg['domain_knowledge']['train_ratio'])
D = cfg['model']['input_dim'] - 1
initial_gamma = float(cfg['domain_knowledge']['initial_gamma'])
lr_gener = float(cfg['domain_knowledge']['lr_gener'])
maxiter = int(cfg['domain_knowledge']['maxiter'])

from dmn_knw_gnrtr.generating_array_OD import generate_OD_DO_array
from dmn_knw_gnrtr.generating_array_OD import process_data
from dmn_knw_gnrtr.generating_array_OD import Connect_to_SQL
from dmn_knw_gnrtr.generating_OD_section_pssblty_sparse_array import generating_OD_section_pssblty_sparse_array
from dmn_knw_gnrtr.generating_repeated_or_not_repeated_domain_knowledge import generating_repeated_or_not_repeated_domain_knowledge
from dmn_knw_gnrtr.PYGT_signal_generation_one_hot import PYGT_signal_generation
from dmn_knw_gnrtr.run_PYGT_0917 import run_PYGT
from dmn_knw_gnrtr.test_PYGT_0917 import test_PYGT

for prefix in ("train","test","val"):
    Time_DepartFreDic_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
    OD_path_dic_file_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}OD_path_dic.pkl')
    station_manager_dict_file_path = os.path.join(base_dir, f"{station_manager_dict_name}")
    OD_feature_array_file_path = os.path.join(base_dir,
                                              f'{od_type.upper()}{os.path.sep}{prefix}_OD_feature_array_dic.pkl')
    Date_and_time_OD_path_dic_file_path = os.path.join(base_dir,
                                                       f'{od_type.upper()}{os.path.sep}{prefix}_Date_and_time_OD_path_dic.pkl')
    dir_path = os.path.join(base_dir, od_type.upper())
    train_result_array_OD_or_DO_file_path = os.path.join(dir_path, f'{prefix}_result_array_{od_type.upper()}.pkl')
    normalization_params_file_path = os.path.join(f'data{os.path.sep}suzhou', 'normalization_params.pkl')

    df,x_y_time,sz_conn_to_station_index,station_index_to_sz_conn=Connect_to_SQL(prefix, train_sql, test_sql, val_sql, station_manager_dict_name,
                                                                                 host, user, password, database)
    result_array_OD = generate_OD_DO_array(df, x_y_time, sz_conn_to_station_index, station_index_to_sz_conn, base_dir, prefix, 'od',station_manager_dict_name)
    result_array_DO = generate_OD_DO_array(df, x_y_time, sz_conn_to_station_index, station_index_to_sz_conn, base_dir, prefix, 'do',station_manager_dict_name)

    base_dir = os.path.join(project_root, f"data{os.path.sep}suzhou")
    with open(station_manager_dict_file_path, 'rb') as f:
        station_manager_dict = pickle.load(f)
    output_dir = os.path.join(base_dir, od_type.upper())

    result_array_file_path = os.path.join(dir_path, f'{prefix}_result_array_{od_type.upper()}.pkl')
    with open(result_array_file_path, 'rb') as f:
        intermediate_dic = pickle.load(f, errors='ignore')
        T_N_D_ODs_dic = intermediate_dic['T_N_D_ODs']
        O_data_dic = intermediate_dic['Trip_Production_In_Station_or_Out_Station']
        D_data_dic = intermediate_dic['Trip_Attraction_In_Station_or_Out_Station']

        sorted_times = sorted(T_N_D_ODs_dic.keys())
        T_N_D_OD_array = np.array([T_N_D_ODs_dic[time] for time in sorted_times])
        O_data = np.array([O_data_dic[time] for time in sorted_times])
        D_data = np.array([D_data_dic[time] for time in sorted_times])

        C_data = station_manager_dict['station_distance_matrix']
        q_obs_data = T_N_D_OD_array
        time_steps = len(O_data)
        optimal_gamma, a_fitted, b_fitted, q_predicted_list = fit_trip_generation_model(O_data, D_data, C_data, q_obs_data,
                                                                                        time_steps=time_steps, initial_gamma=initial_gamma, lr_gener=lr_gener, maxiter=maxiter)

        """config = read_cfg_file(args.config_filename)

        config['trip_distribution']['gamma'] = optimal_gamma
        config['trip_distribution']['a'] = a_fitted
        config['trip_distribution']['b'] = b_fitted

        write_cfg_file(args.config_filename, config)"""

        # 保存训练后的参数到 pkl 文件
        pkl_filename = os.path.join(project_root, f"data{os.path.sep}suzhou", 'trip_generation_trained_params.pkl')
        trained_params = {
            'gamma': optimal_gamma,
            'a': a_fitted,
            'b': b_fitted,
        }

        def save_to_pkl(data, filename):
            with open(filename, 'wb') as file:
                pickle.dump(data, file)

        save_to_pkl(trained_params, pkl_filename)

    for str_prdc_attr in ("prdc", "attr"):
        PYGT_signal_generation("OD", base_dir, prefix, station_manager_dict_name, graph_sz_conn_no_name,
                               station_manager_dict_file_path, graph_sz_conn_root,
                               train_result_array_OD_or_DO_file_path,
                               normalization_params_file_path, str_prdc_attr, Using_lat_lng_or_index)
        PYGT_signal_generation("DO", base_dir, prefix, station_manager_dict_name, graph_sz_conn_no_name,
                               station_manager_dict_file_path, graph_sz_conn_root,
                               train_result_array_OD_or_DO_file_path,
                               normalization_params_file_path, str_prdc_attr, Using_lat_lng_or_index)
        if(prefix=="train"):
            run_PYGT(base_dir, prefix, od_type, str_prdc_attr, RGCN_node_features, RGCN_hidden_units, RGCN_output_dim, RGCN_K, lr, epoch_num, train_ratio, Using_lat_lng_or_index)
            test_PYGT(base_dir, prefix, od_type, str_prdc_attr)
        process_data(D, base_dir, prefix, "od", str_prdc_attr, seq_len)
        process_data(D, base_dir, prefix, "do", str_prdc_attr, seq_len)
    generating_OD_section_pssblty_sparse_array(base_dir, prefix, od_type, station_manager_dict_name,
                                               Time_DepartFreDic_file_path, OD_path_dic_file_path,
                                               station_manager_dict_file_path, OD_feature_array_file_path,
                                               Date_and_time_OD_path_dic_file_path)
    generating_repeated_or_not_repeated_domain_knowledge(base_dir, od_type, prefix, repeated_or_not_repeated, seq_len)

