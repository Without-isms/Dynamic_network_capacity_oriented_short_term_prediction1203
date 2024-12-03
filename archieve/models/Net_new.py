from torch import nn
import torch
import torch.nn.functional as F
import random
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../..'))

from models.OD_Net_new import ODNet_new
from archieve.models.DO_Net_new import DONet_new
from archieve.DualInfoTransformer import DualInfoTransformer


class UtilityLayer(nn.Module):
    def __init__(self, input_dim):
        super(UtilityLayer, self).__init__()
        # Initialize the weights for utility calculation
        # Initial weight is utility = -station_count - 2*transfer_count
        self.weights = nn.Parameter(torch.tensor([-1.0, -2.0]))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.weights.device)
        # x is expected to be of shape (origins, destinations, paths, 2)
        utility = torch.matmul(x, self.weights)
        return utility


class LogitLayer(nn.Module):
    def __init__(self):
        super(LogitLayer, self).__init__()

    def forward(self, utility):
        # utility is expected to be of shape (origins, destinations, paths)
        exp_utility = torch.exp(utility)
        probability = exp_utility / exp_utility.sum(dim=-1, keepdim=True)
        return probability


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class Net_new(torch.nn.Module):

    def __init__(self, cfg, logger):
        super(Net_new, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.additional_section_feature_dim = 5
        self.additional_frequency_feature_dim = 6

        # Updated input dimensions to include additional features
        self.num_finished_input_dim = cfg['model']['input_dim'] + self.additional_section_feature_dim + self.additional_frequency_feature_dim
        self.num_unfinished_input_dim = cfg['model']['input_dim'] + self.additional_section_feature_dim + self.additional_frequency_feature_dim

        self.OD = ODNet_new(cfg, logger)
        self.DO = DONet_new(cfg, logger)

        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']

        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.head = cfg['model'].get('head', 4)
        self.d_channel = cfg['model'].get('channel', 512)

        self.use_curriculum_learning = self.cfg['model']['use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(data=[self.cfg['model']['cl_decay_steps']])
        self.use_input = cfg['model'].get('use_input', True)

        self.mediate_activation = nn.PReLU(self.num_units)

        self.global_step = 0

        self.encoder_first_interact = DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)

        self.decoder_first_interact = DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)

        self.encoder_second_interact = nn.ModuleList([DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)
            for _ in range(self.num_rnn_layers - 1)])

        self.decoder_second_interact = nn.ModuleList([DualInfoTransformer(
            h=self.head,
            d_nodes=self.num_nodes,
            d_model=self.num_units,
            d_channel=self.d_channel)
            for _ in range(self.num_rnn_layers - 1)])

        # Initialize utility and logit layers
        self.utility_layer = UtilityLayer(input_dim=2)
        self.logit_layer = LogitLayer()

        # Initialize the autoencoder
        self.autoencoder = SimpleAutoencoder(input_dim=self.num_nodes*self.num_nodes, latent_dim=self.additional_section_feature_dim)

    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        try:
            return k / (k + math.exp(step / k))
        except OverflowError:
            return float('inf')

    def encoder_od_do(self, sequences, edge_index, edge_attr=None):
        enc_hiddens_od = [None] * self.num_rnn_layers
        enc_hiddens_do = [None] * self.num_rnn_layers

        finished_hidden_od = None
        long_his_hidden_od = None
        short_his_hidden_od = None

        for t, batch in enumerate(sequences):
            encoder_first_out_od, finished_hidden_od, \
                long_his_hidden_od, short_his_hidden_od, \
                enc_first_hidden_od = self.OD.encoder_first_layer(batch,
                                                                  finished_hidden_od,
                                                                  long_his_hidden_od,
                                                                  short_his_hidden_od,
                                                                  edge_index,
                                                                  edge_attr)

            enc_first_out_do, enc_first_hidden_do = self.DO.encoder_first_layer(batch,
                                                                                enc_hiddens_do[0],
                                                                                edge_index,
                                                                                edge_attr)

            enc_first_interact_info_od, enc_first_interact_info_do = self.encoder_first_interact(
                enc_first_hidden_od,
                enc_first_hidden_do)

            enc_hiddens_od[0] = enc_first_hidden_od + enc_first_interact_info_od
            enc_hiddens_do[0] = enc_first_hidden_do + enc_first_interact_info_do

            enc_mid_out_od = encoder_first_out_od + enc_first_interact_info_od
            enc_mid_out_do = enc_first_out_do + enc_first_interact_info_do

            for index in range(self.num_rnn_layers - 1):
                enc_mid_out_od = self.mediate_activation(enc_mid_out_od)
                enc_mid_out_do = self.mediate_activation(enc_mid_out_do)
                enc_mid_out_od, enc_mid_hidden_od = self.OD.encoder_second_layer(index,
                                                                                 enc_mid_out_od,
                                                                                 enc_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)
                enc_mid_out_do, enc_mid_hidden_do = self.DO.encoder_second_layer(index,
                                                                                 enc_mid_out_do,
                                                                                 enc_hiddens_do[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                enc_mid_interact_info_od, enc_mid_interact_info_do = self.encoder_second_interact[index](
                    enc_mid_hidden_od,
                    enc_mid_hidden_do)

                enc_hiddens_od[index + 1] = enc_mid_hidden_od + enc_mid_interact_info_od
                enc_hiddens_do[index + 1] = enc_mid_hidden_do + enc_mid_interact_info_do

        return enc_hiddens_od, enc_hiddens_do

    def scheduled_sampling(self, out, label, GO):
        if self.training and self.use_curriculum_learning:
            c = random.uniform(0, 1)
            T = self.inverse_sigmoid_scheduler_sampling(
                self.global_step,
                self.cl_decay_steps)
            use_truth_sequence = True if c < T else False
        else:
            use_truth_sequence = False

        if use_truth_sequence:
            # Feed the prev label as the next input
            decoder_input = label
        else:
            # detach from history as input
            decoder_input = out.detach().view(-1, self.num_output_dim)
        if not self.use_input:
            decoder_input = GO.detach()

        return decoder_input

    def decoder_od_do(self, sequences, enc_hiddens_od, enc_hiddens_do, edge_index, edge_attr=None):
        predictions_od = []
        predictions_do = []

        GO_od = torch.zeros(enc_hiddens_od[0].size()[0],
                            self.num_output_dim,
                            dtype=enc_hiddens_od[0].dtype,
                            device=enc_hiddens_od[0].device)
        GO_do = torch.zeros(enc_hiddens_do[0].size()[0],
                            self.num_output_dim,
                            dtype=enc_hiddens_do[0].dtype,
                            device=enc_hiddens_do[0].device)

        dec_input_od = GO_od
        dec_hiddens_od = enc_hiddens_od

        dec_input_do = GO_do
        dec_hiddens_do = enc_hiddens_do

        for t in range(self.horizon):
            dec_first_out_od, dec_first_hidden_od = self.OD.decoder_first_layer(dec_input_od,
                                                                                dec_hiddens_od[0],
                                                                                edge_index,
                                                                                edge_attr)

            dec_first_out_do, dec_first_hidden_do = self.DO.decoder_first_layer(dec_input_do,
                                                                                dec_hiddens_do[0],
                                                                                edge_index,
                                                                                edge_attr)

            dec_first_interact_info_od, dec_first_interact_info_do = self.decoder_first_interact(
                dec_first_hidden_od,
                dec_first_hidden_do)

            dec_hiddens_od[0] = dec_first_hidden_od + dec_first_interact_info_od
            dec_hiddens_do[0] = dec_first_hidden_do + dec_first_interact_info_do
            dec_mid_out_od = dec_first_out_od + dec_first_interact_info_od
            dec_mid_out_do = dec_first_out_do + dec_first_interact_info_do

            for index in range(self.num_rnn_layers - 1):
                dec_mid_out_od = self.mediate_activation(dec_mid_out_od)
                dec_mid_out_do = self.mediate_activation(dec_mid_out_do)
                dec_mid_out_od, dec_mid_hidden_od = self.OD.decoder_second_layer(index,
                                                                                 dec_mid_out_od,
                                                                                 dec_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)
                dec_mid_out_do, dec_mid_hidden_do = self.DO.decoder_second_layer(index,
                                                                                 dec_mid_out_do,
                                                                                 dec_hiddens_do[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                dec_second_interact_info_od, dec_second_interact_info_do = self.decoder_second_interact[index](
                    dec_mid_hidden_od,
                    dec_mid_hidden_do)

                dec_hiddens_od[index + 1] = dec_mid_hidden_od + dec_second_interact_info_od
                dec_hiddens_do[index + 1] = dec_mid_hidden_do + dec_second_interact_info_do
                dec_mid_out_od = dec_mid_out_od + dec_second_interact_info_od
                dec_mid_out_do = dec_mid_out_do + dec_second_interact_info_do

            dec_mid_out_od = dec_mid_out_od.reshape(-1, self.num_units)
            dec_mid_out_do = dec_mid_out_do.reshape(-1, self.num_units)

            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od).view(-1, self.num_nodes, self.num_output_dim)
            dec_mid_out_do = self.DO.output_layer(dec_mid_out_do).view(-1, self.num_nodes, self.num_output_dim)

            predictions_od.append(dec_mid_out_od)
            predictions_do.append(dec_mid_out_do)

            dec_input_od = self.scheduled_sampling(dec_mid_out_od, sequences[t].y_od, GO_od)
            dec_input_do = self.scheduled_sampling(dec_mid_out_do, sequences[t].y_do, GO_do)

        if self.training:
            self.global_step += 1

        return torch.stack(predictions_od).transpose(0, 1), torch.stack(predictions_do).transpose(0, 1)

    def forward(self, sequences, sequences_y):
        extended_sequences = []
        extended_sequences_y =[]

        max_shape = None
        for i, data_batch in enumerate(sequences):
            num_graphs = data_batch.ptr.size(0) - 1
            for j in range(num_graphs):
                start_idx = data_batch.ptr[j]
                end_idx = data_batch.ptr[j + 1]
                x_od_sliced = data_batch.x_od[start_idx:end_idx]
                if max_shape is None:
                    max_shape = x_od_sliced.shape[1]+ self.additional_section_feature_dim + self.additional_frequency_feature_dim
                else:
                    max_shape = max(max_shape, x_od_sliced.shape[1] + self.additional_section_feature_dim+ self.additional_frequency_feature_dim)

        for i, data_batch in enumerate(sequences):
            data_batch_y=sequences_y[i]
            num_graphs = data_batch.ptr.size(0) - 1
            import copy
            data_batch_new = copy.deepcopy(data_batch)
            data_batch_y_new=copy.deepcopy(data_batch_y)

            if data_batch_new.x_od.shape[1] < max_shape:
                padding_size = max_shape - data_batch_new.x_od.shape[1]
                data_batch_new.x_od = F.pad(data_batch_new.x_od, (0, padding_size), "constant", 0)

            if data_batch_new.x_do.shape[1] < max_shape:
                padding_size = max_shape - data_batch_new.x_do.shape[1]
                data_batch_new.x_do = F.pad(data_batch_new.x_do, (0, padding_size), "constant", 0)

            if data_batch_new.history.shape[1] < max_shape:
                padding_size = max_shape - data_batch_new.history.shape[1]
                data_batch_new.history = F.pad(data_batch_new.history, (0, padding_size), "constant", 0)

            if data_batch_new.yesterday.shape[1] < max_shape:
                padding_size = max_shape - data_batch_new.yesterday.shape[1]
                data_batch_new.yesterday = F.pad(data_batch_new.yesterday, (0, padding_size), "constant", 0)

            for j in range(num_graphs):
                # 确定每个图的起始和结束索引
                start_idx = data_batch.ptr[j]
                end_idx = data_batch.ptr[j + 1]
                edge_index=(data_batch.edge_index[:, (data_batch.edge_index[0] >= start_idx) & (
                            data_batch.edge_index[0] < end_idx)] - start_idx)
                edge_attr=data_batch.edge_attr[
                    (data_batch.edge_index[0] >= start_idx) & (data_batch.edge_index[0] < end_idx)]
                x_do_sliced=data_batch.x_do[start_idx:end_idx]
                x_od_sliced=data_batch.x_od[start_idx:end_idx]
                unfinished_sliced=data_batch.unfinished[start_idx:end_idx]
                history_sliced=data_batch.history[start_idx:end_idx]
                yesterday_sliced=data_batch.yesterday[start_idx:end_idx]
                PINN_od_features_sliced=data_batch.PINN_od_features[start_idx:end_idx]
                PINN_do_features_sliced=data_batch.PINN_do_features[start_idx:end_idx]
                OD_feature_array_sliced=data_batch.OD_feature_array[start_idx:end_idx]
                OD_path_compressed_array_sliced=data_batch.OD_path_compressed_array[start_idx:end_idx]
                Time_DepartFreDic_Array_sliced=data_batch.Time_DepartFreDic_Array[start_idx:end_idx]
                repeated_sparse_tensors_sliced=data_batch.repeated_sparse_tensors[j]

                """
                x_sample = data_batch.x_od[data_batch.batch == 0]
                """

                if isinstance(OD_feature_array_sliced, np.ndarray):
                    OD_feature_array_sliced = torch.from_numpy(OD_feature_array_sliced).float().to(self.utility_layer.weights.device)
                if isinstance(OD_path_compressed_array_sliced, np.ndarray):
                    OD_path_compressed_array_sliced = torch.from_numpy(OD_path_compressed_array_sliced).float().to(
                        self.utility_layer.weights.device)

                # Step 1: Compute utility
                utility = self.utility_layer(OD_feature_array_sliced)  # shape: (origins, destinations, paths)

                # Step 2: Compute possibility using Logit
                OD_path_possibility = self.logit_layer(utility)  # shape: (origins, destinations, paths)

                # Step 3: Compute OD_section_possibility
                num_origins, num_destinations, num_paths, _ = OD_path_compressed_array_sliced.size()
                num_stations = self.num_nodes

                # 预分配存储邻接矩阵的张量
                OD_section_possibility = torch.ones(num_origins, num_stations, num_stations,
                                                     device=self.utility_layer.weights.device)

                # OD_section_possibility_list = []


                """
                aggregated_OD_section_possibility_list = []

                for i in range(154):
                    # row_result = []
                    print("i" + i)
                    aggregated_row_result = torch.zeros(154, 154)
                    for j in range(154):
                        print("j" + j)
                        sub_result = torch.zeros(154, 154)
                        for k in range(3):
                            # 获取对应的元素
                            sparse_matrix = repeated_sparse_tensors_sliced[i][j][k]

                            # 对应的稠密张量片段
                            possibility_matrix = OD_path_possibility[i, j, k].repeat(154, 154).to('cuda:0')

                            if sparse_matrix == []:
                                # 如果不是稀疏矩阵，生成一个全零矩阵，形状与预期的结果相同
                                result_shape = (154, possibility_matrix.size(1))
                                result_matrix = torch.zeros(result_shape, dtype=torch.int8)
                            else:
                                # 如果是稀疏矩阵，执行稀疏矩阵与稠密矩阵的乘法
                                sparse_matrix = sparse_matrix[0].to('cuda:0')
                                sparse_matrix = sparse_matrix.to('cuda:0').float()
                                result_matrix = torch.sparse.mm(sparse_matrix, possibility_matrix)

                            # 将结果添加到 sub_result 列表中
                            sub_result = sub_result.to('cuda:0')
                            result_matrix = result_matrix.to('cuda:0')
                            sub_result = sub_result + result_matrix
                            aggregated_row_result = aggregated_row_result.to('cuda:0')
                            result_matrix = result_matrix.to('cuda:0')
                            aggregated_row_result = (aggregated_row_result + result_matrix).to('cuda:0')

                            # 将 sub_result 列表转换为张量，并添加到 row_result 中
                        # row_result.append(torch.stack(sub_result).to('cuda:0'))

                    # 将 row_result 列表转换为张量，并添加到最终结果列表中
                    # OD_section_possibility_list.append(torch.stack(row_result).to('cuda:0'))
                    aggregated_OD_section_possibility_list.append(aggregated_row_result.to('cuda:0'))

                # 将最终结果列表转换为一个 PyTorch 张量
                # OD_section_possibility = torch.stack(OD_section_possibility_list).to('cuda:0')
                aggregated_OD_section_possibility = torch.stack(aggregated_OD_section_possibility_list).to('cuda:0'
                """


                """temp_adj_path = torch.zeros(num_stations, num_stations, device=self.utility_layer.weights.device)
                for i in range(num_origins):
                    print("num_origins" + str(i))
                    for j in range(num_destinations):
                        # print("num_destinations" + str(j))
                        temp_adj_OD = torch.zeros(num_stations, num_stations, device=self.utility_layer.weights.device)
                        for k in range(num_paths):
                            path = data_batch.OD_path_compressed_array[i, j, k]
                            temp_adj_path.zero_()
                            path_possibility = possibility[i, j, k].item()
                            for l in range(path.size(0) - 1):
                                start_station = int(path[l].item())
                                if (start_station == -1):
                                    break
                                end_station = int(path[l + 1].item())
                                temp_adj_path[start_station, end_station] = path_possibility
                            temp_adj_OD += temp_adj_path
                        OD_section_possibility[i, j] = temp_adj_OD"""

                # Step 4: Process OD_section_possibility through autoencoder
                aggregated_od_section = OD_section_possibility  # 聚合目的地，得到 [num_origins, num_stations, num_stations]
                flattened_od_section = aggregated_od_section.view(num_origins, -1)  # 展平为 [num_origins, num_stations * num_stations]

                """
                                flattened_od_section = aggregated_OD_section_possibility.view(num_origins, -1)  # 展平为 [num_origins, num_stations * num_stations]
                """
                latent_features, _ = self.autoencoder(flattened_od_section)
                x_od_combined = torch.cat((x_od_sliced, latent_features), dim=1)
                x_od_combined = torch.cat((x_od_combined, Time_DepartFreDic_Array_sliced), dim=1)
                x_do_combined = torch.cat((x_do_sliced, latent_features), dim=1)
                x_do_combined = torch.cat((x_do_combined, Time_DepartFreDic_Array_sliced), dim=1)
                history_combined = torch.cat((history_sliced, latent_features), dim=1)
                history_combined = torch.cat((history_combined, Time_DepartFreDic_Array_sliced), dim=1)
                yesterday_combined = torch.cat((yesterday_sliced, latent_features), dim=1)
                yesterday_combined = torch.cat((yesterday_combined, Time_DepartFreDic_Array_sliced), dim=1)

                current_shape = data_batch_new.x_od[start_idx:end_idx].shape[1]
                target_shape = x_od_combined.shape[1]

                if current_shape < target_shape:
                    # 如果x_od_combined维度小于目标维度，用0进行填充
                    padding_size = target_shape - current_shape
                    x_od_combined = F.pad(x_od_combined, (0, padding_size), "constant", 0)
                    x_do_combined = F.pad(x_do_combined, (0, padding_size), "constant", 0) if x_do_combined.shape[
                                                                                                  1] < target_shape else x_do_combined[
                                                                                                                         :,
                                                                                                                         :target_shape]
                    history_combined = F.pad(history_combined, (0, padding_size), "constant", 0) if \
                    history_combined.shape[1] < target_shape else history_combined[:, :target_shape]
                    yesterday_combined = F.pad(yesterday_combined, (0, padding_size), "constant", 0) if \
                    yesterday_combined.shape[1] < target_shape else yesterday_combined[:, :target_shape]

                data_batch_new.x_od[start_idx:end_idx] = x_od_combined
                data_batch_new.x_do[start_idx:end_idx] = x_do_combined
                data_batch_new.unfinished[start_idx:end_idx]
                data_batch_new.history[start_idx:end_idx] = history_combined
                data_batch_new.yesterday[start_idx:end_idx] = yesterday_combined

            extended_sequences.append(data_batch_new)
            extended_sequences_y.append(data_batch_y_new)

        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()

        enc_hiddens_od, enc_hiddens_do = self.encoder_od_do(extended_sequences,
                                                            edge_index=edge_index,
                                                            edge_attr=edge_attr)
        predictions_od, predictions_do = self.decoder_od_do(extended_sequences_y,
                                                            enc_hiddens_od,
                                                            enc_hiddens_do,
                                                            edge_index=edge_index,
                                                            edge_attr=edge_attr)

        return predictions_od, predictions_do