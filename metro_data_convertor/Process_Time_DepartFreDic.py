import pickle
import numpy as np
import os

from metro_data_convertor.Find_project_root import Find_project_root


def Process_Time_DepartFreDic(Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path, prefix):
    with open(Time_DepartFreDic_file_path, 'rb') as file:
        Time_DepartFreDic = pickle.load(file)

    unique_stations = set()
    unique_lines = set()

    for DF_of_same_time in Time_DepartFreDic.values():
        if DF_of_same_time=={}:
            continue
        for route in DF_of_same_time.keys():
            start_info = DF_of_same_time[route]['start_station']
            terminal_info = DF_of_same_time[route]['terminal_station']
            unique_stations.add((start_info['name'], start_info['index']))
            unique_stations.add((terminal_info['name'], terminal_info['index']))
            unique_lines.add(DF_of_same_time[route]['section_line'])

    sorted_stations = sorted(unique_stations, key=lambda x: x[1])
    station_names = [station[0] for station in sorted_stations]
    station_indices = {name: idx for idx, (name, _) in enumerate(sorted_stations)}

    sorted_lines = sorted(unique_lines)
    line_indices = {line: idx for idx, line in enumerate(sorted_lines)}

    result_dict = {}
    for timestamp, sections in Time_DepartFreDic.items():
        frequency_array = np.zeros((len(station_names), len(sorted_lines)))

        if sections == {}:
            result_dict[timestamp] = frequency_array
            continue
        for route, info in sections.items():
            start_idx = station_indices[info['start_station']['name']]
            line_idx = line_indices[info['section_line']]
            if info['depart_freq']==0:
                frequency_array[start_idx, line_idx] = 0
            else:
                freq_minutes = info['depart_freq'].hour * 60 + info['depart_freq'].minute
                frequency_array[start_idx, line_idx] = freq_minutes
        result_dict[timestamp] = frequency_array

    with open(Time_DepartFreDic_Array_file_path, 'wb') as f:
        pickle.dump(result_dict, f)

    sample_timestamp, sample_frequency_array = next(iter(result_dict.items()))
    print("Station name:", station_names)
    print("Lines:", sorted_lines)
    print("Sample timestamp:", sample_timestamp)
    print("Sample frequency array:\n", sample_frequency_array)

"""project_root = Find_project_root()
Time_DepartFreDic_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
Time_DepartFreDic_Array_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic_Array.pkl')
Process_Time_DepartFreDic(Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path)"""

