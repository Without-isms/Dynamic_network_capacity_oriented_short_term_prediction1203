# part of this code are copied from DCRNN
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch_geometric.data import Batch, Data
# from sklearn.externals import joblib
import tempfile
import gc
import shutil

class DataLoader(object):

    def __init__(self,
                 x_od,
                 y_od,
                 unfinished,
                 history,
                 yesterday,
                 x_do,
                 y_do,
                 xtime,
                 ytime,
                 PINN_od_features,
                 PINN_do_features,
                 PINN_od_additional_features,
                 PINN_do_additional_features,
                 OD_feature_array,
                 OD_path_compressed_array,
                 Time_DepartFreDic_Array,
                 repeated_sparse_tensors,
                 batch_size,
                 pad_with_last_sample=True,
                 shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            ####第二个步骤
            num_padding = (batch_size - (len(x_od) % batch_size)) % batch_size
            x_od_padding = np.repeat(x_od[-1:], num_padding, axis=0)
            x_do_padding = np.repeat(x_do[-1:], num_padding, axis=0)
            xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
            y_od_padding = np.repeat(y_od[-1:], num_padding, axis=0)
            y_do_padding = np.repeat(y_do[-1:], num_padding, axis=0)
            ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)
            PINN_od_features_padding = np.repeat(PINN_od_features[-1:], num_padding, axis=0)
            PINN_do_features_padding = np.repeat(PINN_do_features[-1:], num_padding, axis=0)
            PINN_od_additional_features_padding = np.repeat(PINN_od_additional_features[-1:], num_padding, axis=0)
            PINN_do_additional_features_padding= np.repeat(PINN_do_additional_features[-1:], num_padding, axis=0)
            OD_feature_array_padding = np.repeat(OD_feature_array[-1:], num_padding, axis=0)
            OD_path_compressed_array_padding = np.repeat(OD_path_compressed_array[-1:], num_padding, axis=0)
            Time_DepartFreDic_Array_padding = np.repeat(Time_DepartFreDic_Array[-1:], num_padding, axis=0)
            repeated_sparse_padding = [repeated_sparse_tensors[-1]] * num_padding

            x_od = np.concatenate([x_od, x_od_padding], axis=0)
            x_do = np.concatenate([x_do, x_do_padding], axis=0)
            xtime = np.concatenate([xtime, xtime_padding], axis=0)
            y_od = np.concatenate([y_od, y_od_padding], axis=0)
            y_do = np.concatenate([y_do, y_do_padding], axis=0)
            ytime = np.concatenate([ytime, ytime_padding], axis=0)
            PINN_od_features = np.concatenate([PINN_od_features, PINN_od_features_padding], axis=0)
            PINN_do_features = np.concatenate([PINN_do_features, PINN_do_features_padding], axis=0)
            PINN_od_additional_features = np.concatenate([PINN_od_additional_features, PINN_od_additional_features_padding], axis=0)
            PINN_do_additional_features = np.concatenate(
                [PINN_do_additional_features, PINN_do_additional_features_padding], axis=0)
            PINN_od_features = np.concatenate([PINN_od_features, PINN_od_features_padding], axis=0)
            PINN_do_features = np.concatenate([PINN_do_features, PINN_do_features_padding], axis=0)
            OD_feature_array = np.concatenate([OD_feature_array, OD_feature_array_padding], axis=0)
            OD_path_compressed_array = np.concatenate([OD_path_compressed_array, OD_path_compressed_array_padding], axis=0)
            Time_DepartFreDic_Array = np.concatenate([Time_DepartFreDic_Array, Time_DepartFreDic_Array_padding], axis=0)
            repeated_sparse_tensors = repeated_sparse_tensors + repeated_sparse_padding

            unfi_num_padding = (batch_size - (len(unfinished) % batch_size)) % batch_size
            unfinished_padding = np.repeat(unfinished[-1:], unfi_num_padding, axis=0)
            unfinished = np.concatenate([unfinished, unfinished_padding], axis=0)

            history_num_padding = (batch_size - (len(history) % batch_size)) % batch_size
            history_padding = np.repeat(history[-1:], history_num_padding, axis=0)
            history = np.concatenate([history, history_padding], axis=0)

            yesterday_num_padding = (batch_size - (len(yesterday) % batch_size)) % batch_size
            yesterday_padding = np.repeat(yesterday[-1:], yesterday_num_padding, axis=0)
            yesterday = np.concatenate([yesterday, yesterday_padding], axis=0)
        self.size = len(x_od)
        self.num_batch = int(self.size // self.batch_size)
        # if shuffle:
        #     permutation = np.random.permutation(self.size)
        #     x_do, y_do = x_do[permutation], y_do[permutation]
        #     x_od, y_od = x_od[permutation], y_od[permutation]
        #     xtime, ytime = xtime[permutation], ytime[permutation]
        #     unfinished = unfinished[permutation]
        #     history = history[permutation]
        #     yesterday = yesterday[permutation]
        self.x_do = x_do
        self.y_do = y_do
        self.x_od = x_od
        self.y_od = y_od
        self.unfinished = unfinished
        self.history = history
        self.yesterday = yesterday
        self.xtime = xtime
        self.ytime = ytime
        self.PINN_od_features = PINN_od_features
        self.PINN_do_features = PINN_do_features
        self.PINN_od_additional_features = PINN_od_additional_features
        self.PINN_do_additional_features = PINN_do_additional_features
        self.OD_feature_array = OD_feature_array
        self.OD_path_compressed_array = OD_path_compressed_array
        self.Time_DepartFreDic_Array = Time_DepartFreDic_Array
        self.repeated_sparse_tensors = repeated_sparse_tensors

    def shuffle(self):
        ####第三个步骤
        """
        permutation = np.random.permutation(self.size)
        x_do, y_do = self.x_do[permutation], self.y_do[permutation]
        x_od, y_od = self.x_od[permutation], self.y_od[permutation]
        xtime, ytime = self.xtime[permutation], self.ytime[permutation]
        PINN_od_features, PINN_do_features = self.PINN_od_features[permutation], self.PINN_do_features[permutation]
        PINN_od_additional_features, PINN_do_additional_features = self.PINN_od_additional_features[permutation], self.PINN_do_additional_features[permutation]
        OD_feature_array = self.OD_feature_array[permutation]
        OD_path_compressed_array = self.OD_path_compressed_array[permutation]
        unfinished = self.unfinished[permutation]
        history = self.history[permutation]
        yesterday = self.yesterday[permutation]


        self.x_do = x_do
        self.y_do = y_do
        self.x_od = x_od
        self.y_od = y_od
        self.unfinished = unfinished
        self.history = history
        self.yesterday = yesterday
        self.xtime = xtime
        self.ytime = ytime
        self.PINN_od_features = PINN_od_features
        self.PINN_do_features = PINN_do_features
        self.PINN_od_additional_features = PINN_od_additional_features
        self.PINN_do_additional_features = PINN_do_additional_features
        self.OD_feature_array = OD_feature_array
        self.OD_path_compressed_array = OD_path_compressed_array
        """

        # Assuming self.size is the size of the arrays
        def apply_permutation_in_chunks_memmap(array, permutation, chunk_size, filename, temp_dir):
            shape = array.shape
            dtype = array.dtype

            # 将文件名和新的临时目录路径结合
            filename = os.path.join(temp_dir, filename)

            # 手动创建文件并初始化大小
            with open(filename, 'wb') as f:
                f.write(b'\x00' * np.product(shape) * np.dtype(dtype).itemsize)

            # 创建内存映射的空数组，用于存储结果
            mmapped_result = np.memmap(filename, dtype=dtype, mode='r+', shape=shape)

            # 分块应用排列
            for start_idx in range(0, len(permutation), chunk_size):
                end_idx = min(start_idx + chunk_size, len(permutation))
                # print(f'Writing from {start_idx} to {end_idx}')  # 添加调试信息
                mmapped_result[start_idx:end_idx] = array[permutation[start_idx:end_idx]]
                # print(f'Block {start_idx} to {end_idx} written')

            # 确保所有更改被写入磁盘
            mmapped_result.flush()

            # 强制刷新文件系统缓存
            with open(filename, 'r+') as f:
                os.fsync(f.fileno())

            return mmapped_result

        # 创建一个新的临时目录
        temp_dir = tempfile.mkdtemp()

        try:
            # 定义合适的块大小
            chunk_size = 40  # 根据内存限制调整此值

            # 生成排列顺序
            permutation = np.random.permutation(self.size)

            # 对每个数组应用内存映射和逐块处理
            self.x_do = apply_permutation_in_chunks_memmap(self.x_do, permutation, chunk_size, 'x_do.dat', temp_dir)
            self.y_do = apply_permutation_in_chunks_memmap(self.y_do, permutation, chunk_size, 'y_do.dat', temp_dir)
            self.x_od = apply_permutation_in_chunks_memmap(self.x_od, permutation, chunk_size, 'x_od.dat', temp_dir)
            self.y_od = apply_permutation_in_chunks_memmap(self.y_od, permutation, chunk_size, 'y_od.dat', temp_dir)
            self.xtime = apply_permutation_in_chunks_memmap(self.xtime, permutation, chunk_size, 'xtime.dat', temp_dir)
            self.ytime = apply_permutation_in_chunks_memmap(self.ytime, permutation, chunk_size, 'ytime.dat', temp_dir)
            self.PINN_od_features = apply_permutation_in_chunks_memmap(self.PINN_od_features, permutation, chunk_size,
                                                                       'PINN_od_features.dat', temp_dir)
            self.PINN_do_features = apply_permutation_in_chunks_memmap(self.PINN_do_features, permutation, chunk_size,
                                                                       'PINN_do_features.dat', temp_dir)
            self.PINN_od_additional_features = apply_permutation_in_chunks_memmap(self.PINN_od_additional_features,
                                                                                  permutation, chunk_size,
                                                                                  'PINN_od_additional_features.dat',
                                                                                  temp_dir)
            self.PINN_do_additional_features = apply_permutation_in_chunks_memmap(self.PINN_do_additional_features,
                                                                                  permutation, chunk_size,
                                                                                  'PINN_do_additional_features.dat',
                                                                                  temp_dir)
            self.OD_feature_array = apply_permutation_in_chunks_memmap(self.OD_feature_array, permutation, chunk_size,
                                                                       'OD_feature_array.dat', temp_dir)
            self.OD_path_compressed_array = apply_permutation_in_chunks_memmap(self.OD_path_compressed_array,
                                                                               permutation,
                                                                               chunk_size,
                                                                               'OD_path_compressed_array.dat', temp_dir)
            self.unfinished = apply_permutation_in_chunks_memmap(self.unfinished, permutation, chunk_size,
                                                                 'unfinished.dat', temp_dir)
            self.history = apply_permutation_in_chunks_memmap(self.history, permutation, chunk_size, 'history.dat',
                                                              temp_dir)
            self.yesterday = apply_permutation_in_chunks_memmap(self.yesterday, permutation, chunk_size,
                                                                'yesterday.dat', temp_dir)
            self.Time_DepartFreDic_Array = apply_permutation_in_chunks_memmap(self.Time_DepartFreDic_Array, permutation, chunk_size,
                                                                'Time_DepartFreDic_Array.dat', temp_dir)

            self.repeated_sparse_tensors = [self.repeated_sparse_tensors[i] for i in permutation]

        finally:
            # 删除整个临时目录及其中的所有文件
            # shutil.rmtree(temp_dir)
            print(f'Save to temporary directory: {temp_dir}')

    def get_iterator(self):
        self.current_ind = 0

        ####第四个步骤
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_do_i = self.x_do[start_ind:end_ind, ...]
                y_do_i = self.y_do[start_ind:end_ind, ...]
                x_od_i = self.x_od[start_ind:end_ind, ...]
                y_od_i = self.y_od[start_ind:end_ind, ...]
                unfinished_i = self.unfinished[start_ind:end_ind, ...]
                history_i = self.history[start_ind:end_ind, ...]
                yesterday_i = self.yesterday[start_ind:end_ind, ...]
                xtime_i = self.xtime[start_ind:end_ind, ...]
                ytime_i = self.ytime[start_ind:end_ind, ...]
                PINN_od_features_i = self.PINN_od_features[start_ind:end_ind, ...]
                PINN_do_features_i = self.PINN_do_features[start_ind:end_ind, ...]
                PINN_od_additional_features_i = self.PINN_od_additional_features[start_ind:end_ind, ...]
                PINN_do_additional_features_i = self.PINN_do_additional_features[start_ind:end_ind, ...]
                OD_feature_array_features_i = self.OD_feature_array[start_ind:end_ind, ...]
                OD_path_compressed_array_i = self.OD_path_compressed_array[start_ind:end_ind, ...]
                Time_DepartFreDic_Array_i = self.Time_DepartFreDic_Array[start_ind:end_ind, ...]
                repeated_sparse_tensors_i = self.repeated_sparse_tensors[start_ind:end_ind]
                yield (x_od_i, y_od_i, x_do_i, y_do_i, unfinished_i, history_i, yesterday_i, xtime_i, ytime_i,
                       PINN_od_features_i,PINN_do_features_i,PINN_od_additional_features_i,PINN_do_additional_features_i,
                       OD_feature_array_features_i,OD_path_compressed_array_i,Time_DepartFreDic_Array_i,
                       repeated_sparse_tensors_i)  # 注意顺序
                self.current_ind += 1
        return _wrapper()

class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def load_dataset(dataset_dir,
                    do_dataset_dir,
                    batch_size,
                    test_batch_size=None,
                    scaler_axis=(0,
                                 1,
                                 2,
                                 3),
                    **kwargs):
    data = {}



    for category in ['train', 'val', 'test']:
        ####第一个步骤
        cat_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
        history_data = load_pickle(os.path.join(dataset_dir, category + '_history_long.pkl'))
        yesterday_data = load_pickle(os.path.join(dataset_dir, category + '_history_short.pkl'))
        do_data = load_pickle(os.path.join(do_dataset_dir, category + '_do.pkl'))
        PINN_data_od = load_pickle(os.path.join(dataset_dir, category + '_OD_signal_dict_array.pkl'))
        PINN_data_do = load_pickle(os.path.join(do_dataset_dir, category + '_DO_signal_dict_array.pkl'))
        OD_feature_array = load_pickle(os.path.join(dataset_dir, category + '_OD_feature_array.pkl'))
        OD_path_compressed_array = load_pickle(os.path.join(dataset_dir, category + '_OD_path_compressed_array.pkl'))
        Time_DepartFreDic_Array = load_pickle(os.path.join(dataset_dir, category + '_repeated_Time_DepartFre_Array.pkl'))
        repeated_sparse_tensors = torch.load(os.path.join(dataset_dir, category + '_repeated_sparse_tensors.pt'))

        data['x_' + category] = cat_data['finished']
        data['unfinished_' + category] = cat_data['unfinished']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']
        data['do_x_'+ category] = do_data['finished']
        data['do_y_' + category] = do_data['y']
        data['PINN_od_features_' + category] = np.array(PINN_data_od["features"])
        data['PINN_od_additional_features_' + category] = PINN_data_od["additional_feature"]
        data['PINN_do_features_' + category] = PINN_data_do["features"]
        data['PINN_do_additional_features_' + category] = PINN_data_do["additional_feature"]
        data['OD_feature_array_' + category] = OD_feature_array
        data['OD_path_compressed_array_' + category] = OD_path_compressed_array
        data['Time_DepartFreDic_Array_' + category] = Time_DepartFreDic_Array
        data['repeated_sparse_tensors_' + category] = repeated_sparse_tensors

        #data['do_x_'+ category] = do_data['do_x']
        #data['do_y_' + category] = do_data['do_y']

        his_sum = np.sum(history_data['history'], axis=-1)
        history_distribution = np.nan_to_num(np.divide(history_data['history'], np.expand_dims(his_sum, axis=-1)))
        data['history_' + category] = np.multiply(history_distribution, cat_data['unfinished'])

        yesterday_sum = np.sum(yesterday_data['history'], axis=-1)
        yesterday_distribution = np.nan_to_num(np.divide(yesterday_data['history'], np.expand_dims(yesterday_sum, axis=-1)))
        data['yesterday_' + category] = np.multiply(yesterday_distribution, cat_data['unfinished'])

    scaler = StandardScaler(mean=data['x_train'].mean(axis=scaler_axis),
                            std=data['x_train'].std(axis=scaler_axis))
    do_scaler = StandardScaler(mean=data['do_x_train'].mean(axis=scaler_axis),
                                    std=data['do_x_train'].std(axis=scaler_axis))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
        data['do_x_' + category] = do_scaler.transform(data['do_x_' + category])
        data['do_y_' + category] = do_scaler.transform(data['do_y_' + category])
        data['unfinished_' + category] = scaler.transform(data['unfinished_' + category])
        data['history_' + category] = scaler.transform(data['history_' + category])
        data['yesterday_' + category] = scaler.transform(data['yesterday_' + category])

    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      data['unfinished_train'],
                                      data['history_train'],
                                      data['yesterday_train'],
                                      data['do_x_train'],
                                      data['do_y_train'],
                                      data['xtime_train'],
                                      data['ytime_train'],
                                      data['PINN_od_features_train'],
                                      data['PINN_do_features_train'],
                                      data['PINN_od_additional_features_train'],
                                      data['PINN_do_additional_features_train'],
                                      data['OD_feature_array_train'],
                                      data['OD_path_compressed_array_train'],
                                      data['Time_DepartFreDic_Array_train'],
                                      data['repeated_sparse_tensors_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    data['unfinished_val'],
                                    data['history_val'],
                                    data['yesterday_val'],
                                    data['do_x_val'],
                                    data['do_y_val'],
                                    data['xtime_val'],
                                    data['ytime_val'],
                                    data['PINN_od_features_val'],
                                    data['PINN_do_features_val'],
                                    data['PINN_od_additional_features_val'],
                                    data['PINN_do_additional_features_val'],
                                    data['OD_feature_array_val'],
                                    data['OD_path_compressed_array_val'],
                                    data['Time_DepartFreDic_Array_val'],
                                    data['repeated_sparse_tensors_val'],
                                    test_batch_size,
                                    shuffle=False)

    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     data['unfinished_test'],
                                     data['history_test'],
                                     data['yesterday_test'],
                                     data['do_x_test'],
                                     data['do_y_test'],
                                     data['xtime_test'],
                                     data['ytime_test'],
                                     data['PINN_od_features_test'],
                                     data['PINN_do_features_test'],
                                     data['PINN_od_additional_features_test'],
                                     data['PINN_do_additional_features_test'],
                                     data['OD_feature_array_test'],
                                     data['OD_path_compressed_array_test'],
                                     data['Time_DepartFreDic_Array_test'],
                                     data['repeated_sparse_tensors_test'],
                                     test_batch_size,
                                     shuffle=False)
    data['scaler'] = scaler
    data['do_scaler'] = do_scaler

    return data

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class SimpleBatch(list):

    def to(self, device):
        for ele in self:
            ele.to(device)
        return self

def collate_wrapper(x_od, y_od, x_do, y_do, unfinished, history, yesterday, PINN_od_features, PINN_do_features,
                    PINN_od_additional_features,PINN_do_additional_features, OD_feature_array, OD_path_compressed_array,
                    Time_DepartFreDic_Array, repeated_sparse_tensors, edge_index, edge_attr, seq_len, horizon, device, return_y=True):
    x_od = torch.tensor(x_od, dtype=torch.float, device=device)
    y_od = torch.tensor(y_od, dtype=torch.float, device=device)
    x_od = x_od.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_od_T_first = y_od.transpose(dim0=1, dim1=0)

    x_do = torch.tensor(x_do, dtype=torch.float, device=device)
    y_do = torch.tensor(y_do, dtype=torch.float, device=device)
    x_do = x_do.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_do_T_first = y_do.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)

    unfinished = torch.tensor(unfinished, dtype=torch.float, device=device)
    unfinished = unfinished.transpose(dim0=1, dim1=0)

    history = torch.tensor(history, dtype=torch.float, device=device)
    history = history.transpose(dim0=1, dim1=0)

    yesterday = torch.tensor(yesterday, dtype=torch.float, device=device)
    yesterday = yesterday.transpose(dim0=1, dim1=0)

    transposed_repeated_sparse_tensors = list(map(list, zip(*repeated_sparse_tensors)))

    PINN_od_features = torch.tensor(PINN_od_features, dtype=torch.float, device=device)
    PINN_do_features = torch.tensor(PINN_do_features, dtype=torch.float, device=device)
    PINN_od_additional_features = torch.tensor(PINN_od_additional_features, dtype=torch.float, device=device)
    PINN_do_additional_features = torch.tensor(PINN_do_additional_features, dtype=torch.float, device=device)
    PINN_od_features_T_first = PINN_od_features.transpose(dim0=1, dim1=0) # (T, N, num_nodes, num_features)
    PINN_do_features_T_first = PINN_do_features.transpose(dim0=1, dim1=0) # (T, N, num_nodes, num_features)
    PINN_od_additional_features_T_first = PINN_od_additional_features.transpose(dim0=1, dim1=0) # (T, N, num_nodes, num_features)
    PINN_do_additional_features_T_first = PINN_do_additional_features.transpose(dim0=1, dim1=0) # (T, N, num_nodes, num_features)

    OD_feature_array = torch.tensor(OD_feature_array, dtype=torch.float, device=device)
    OD_path_compressed_array = torch.tensor(OD_path_compressed_array, dtype=torch.float, device=device)
    OD_feature_array_T_first = OD_feature_array.transpose(dim0=1, dim1=0)
    OD_path_compressed_array_T_first = OD_path_compressed_array.transpose(dim0=1, dim1=0)

    Time_DepartFreDic_Array = torch.tensor(Time_DepartFreDic_Array, dtype=torch.float, device=device)
    Time_DepartFreDic_Array_T_first = Time_DepartFreDic_Array.transpose(dim0=1, dim1=0)

    edge_index = torch.tensor(edge_index, device=device)
    edge_attr = torch.tensor(edge_attr, device=device)

    """
    print(f"x_od device: {x_od.device}")
    print(f"y_od device: {y_od.device}")
    print(f"x_do device: {x_do.device}")
    print(f"y_do device: {y_do.device}")
    print(f"unfinished device: {unfinished.device}")
    print(f"history device: {history.device}")
    print(f"yesterday device: {yesterday.device}")
    print(f"PINN_od_features device: {PINN_od_features.device}")
    print(f"PINN_do_features device: {PINN_do_features.device}")
    print(f"PINN_od_additional_features device: {PINN_od_additional_features.device}")
    print(f"PINN_do_additional_features device: {PINN_do_additional_features.device}")
    print(f"edge_index device: {edge_index.device}")
    print(f"edge_attr device: {edge_attr.device}")
    """

    #  do not tranpose y_truth
    T = x_od.size()[0]
    H = y_od_T_first.size()[0]
    N = x_od.size()[1]
    # generate batched sequence.
    sequences = []
    sequences_y = []
    for t in range(T - seq_len, T):
        cur_batch_x_od = x_od[t]
        cur_batch_x_do = x_do[t]
        cur_batch_unfinished = unfinished[t]
        cur_batch_history = history[t]
        cur_batch_yesterday = yesterday[t]
        cur_batch_PINN_od_features = PINN_od_features_T_first[t]
        cur_batch_PINN_do_features = PINN_do_features_T_first[t]
        cur_batch_PINN_od_additional_features = PINN_od_additional_features_T_first[t]
        cur_batch_PINN_do_additional_features = PINN_do_additional_features_T_first[t]
        cur_batch_OD_feature_array = OD_feature_array_T_first[t]
        cur_batch_OD_path_compressed_array = OD_path_compressed_array_T_first[t]
        cur_batch_Time_DepartFreDic_Array = Time_DepartFreDic_Array_T_first[t]
        cur_repeated_sparse_tensors=transposed_repeated_sparse_tensors[t]
        """
        from pprint import pprint
        print("transposed_repeated_sparse_tensors")
        pprint(transposed_repeated_sparse_tensors)
        print("cur_repeated_sparse_tensors")
        pprint(cur_repeated_sparse_tensors)
        """

        batch = Batch.from_data_list([
            Data(x_do=cur_batch_x_do[i],
                 x_od=cur_batch_x_od[i],
                 unfinished=cur_batch_unfinished[i],
                 history=cur_batch_history[i],
                 yesterday=cur_batch_yesterday[i],
                 PINN_od_features=cur_batch_PINN_od_features[i],
                 PINN_do_features=cur_batch_PINN_do_features[i],
                 PINN_od_additional_features=cur_batch_PINN_od_additional_features[i],
                 PINN_do_additional_features=cur_batch_PINN_do_additional_features[i],
                 OD_feature_array=cur_batch_OD_feature_array[i],
                 OD_path_compressed_array=cur_batch_OD_path_compressed_array[i],
                 Time_DepartFreDic_Array=cur_batch_Time_DepartFreDic_Array[i],
                 repeated_sparse_tensors=cur_repeated_sparse_tensors[i],
                 edge_index=edge_index,
                 edge_attr=edge_attr) for i in range(N)
            #if not print(f"Length of repeated_sparse_tensors[{i}]: {len(cur_repeated_sparse_tensors[i])}")
        ])
        sequences.append(batch)

    for t in range(H - horizon, H):
        cur_batch_y_od = y_od_T_first[t]
        cur_batch_y_do = y_do_T_first[t]

        batch_y = Batch.from_data_list([
            Data(y_do=cur_batch_y_do[i],
                 y_od=cur_batch_y_od[i]) for i in range(N)
        ])
        sequences_y.append(batch_y)
    if return_y:
        return SimpleBatch(sequences), SimpleBatch(sequences_y), y_od, y_do
    else:
        return SimpleBatch(sequences), SimpleBatch(sequences_y)


def collate_wrapper_multi_branches(x_numpy, y_numpy, edge_index_list, device):
    sequences_multi_branches = []
    for edge_index in edge_index_list:
        sequences, y = collate_wrapper(x_numpy, y_numpy, edge_index, device, return_y=True)
        sequences_multi_branches.append(sequences)

    return sequences_multi_branches, y
