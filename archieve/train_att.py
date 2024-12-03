# encoding:utf-8
import random
import argparse
import time
import yaml
import numpy as np
import torch
import os

from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_
from lib import utils_HIAM_no_DO as utils
from lib import metrics
from lib.utils_HIAM_no_DO import collate_wrapper
from archieve.models.Net_att import Net_att

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--config_filename',
                    default=None,
                    type=str,
                    help='Configuration filename for restoring the model.')
args = argparse.Namespace(config_filename='../data/config/train_sz_dim26_units96_h4c512.yaml')


def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg


def run_model(model, data_iterator, edge_index, edge_attr, device, seq_len, horizon, output_dim):
    """
    return a list of (horizon_i, batch_size, num_nodes, output_dim)
    """
    # while evaluation, we need model.eval and torch.no_grad
    model.eval()
    y_od_pred_list = []
    for _, (x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
            PINN_od_additional_features,OD_feature_array,
            #OD_path_compressed_array,
            Time_DepartFreDic_Array, repeated_sparse_2D_tensors,
            repeated_sparse_3D_tensors,
            # repeated_sparse_4D_tensors, repeated_sparse_5D_tensors
            ) in enumerate(data_iterator):
        y_od = y_od[..., :output_dim]
        sequences, sequences_y, y_od = collate_wrapper(x_od=x_od, y_od=y_od,
                                                       unfinished=unfinished, history=history,
                                                       yesterday=yesterday,
                                                       PINN_od_features=PINN_od_features,
                                                       PINN_od_additional_features=PINN_od_additional_features,
                                                       OD_feature_array=OD_feature_array,
                                                       # OD_path_compressed_array = OD_path_compressed_array,
                                                       Time_DepartFreDic_Array=Time_DepartFreDic_Array,
                                                       repeated_sparse_2D_tensors=repeated_sparse_2D_tensors,
                                                       repeated_sparse_3D_tensors=repeated_sparse_3D_tensors,
                                                       # repeated_sparse_4D_tensors=repeated_sparse_4D_tensors,
                                                       # repeated_sparse_5D_tensors=repeated_sparse_5D_tensors,
                                                       edge_index=edge_index,
                                                       edge_attr=edge_attr,
                                                       device=device,
                                                       seq_len=seq_len,
                                                       horizon=horizon)
        # (T, N, num_nodes, num_out_channels)
        with torch.no_grad():
            y_od_pred = model(sequences, sequences_y)
            if y_od_pred is not None:
                y_od_pred_list.append(y_od_pred.detach().cpu().numpy())
    return y_od_pred_list


def evaluate(model,
             dataset,
             dataset_type,
             edge_index,
             edge_attr,
             device,
             seq_Len,
             horizon,
             output_dim,
             logger,
             detail=True,
             cfg=None,
             format_result=False):
    if detail:
        logger.info('Evaluation_{}_Begin:'.format(dataset_type))

    y_od_preds = run_model(
        model,
        data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        device=device,
        seq_len=seq_Len,
        horizon=horizon,
        output_dim=output_dim)

    evaluate_category = []
    if len(y_od_preds) > 0:
        evaluate_category.append("od")
    results = {}
    for category in evaluate_category:
        if category == 'od':
            y_preds = y_od_preds
            scaler = dataset['scaler']
            gt = dataset['y_{}'.format(dataset_type)]

        y_preds = np.concatenate(y_preds, axis=0)  # concat in batch_size dim.
        mae_list = []
        mape_net_list = []
        rmse_list = []
        mae_sum = 0

        mape_net_sum = 0
        rmse_sum = 0
        logger.info("{}:".format(category))
        horizon = cfg['model']['horizon']
        for horizon_i in range(horizon):
            y_truth = scaler.inverse_transform(
                gt[:, horizon_i, :, :output_dim])

            y_pred = scaler.inverse_transform(
                y_preds[:y_truth.shape[0], horizon_i, :, :output_dim])
            y_pred[y_pred < 0] = 0
            mae = metrics.masked_mae_np(y_pred, y_truth)
            mape_net = metrics.masked_mape_np(y_pred, y_truth)
            rmse = metrics.masked_rmse_np(y_pred, y_truth)
            mae_sum += mae
            mape_net_sum += mape_net
            rmse_sum += rmse
            mae_list.append(mae)

            mape_net_list.append(mape_net)
            rmse_list.append(rmse)

            msg = "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE_net: {:.4f}"
            if detail:
                logger.info(msg.format(horizon_i + 1, mae, rmse, mape_net))
        results['MAE_' + category] = mae_sum / horizon
        results['RMSE_' + category] = rmse_sum / horizon
        results['MAPE_net_' + category] = mape_net_sum / horizon
    if detail:
        logger.info('Evaluation_{}_End:'.format(dataset_type))
    if format_result:
        for i in range(len(mae_list)):
            print('{:.2f}'.format(mae_list[i]))
            print('{:.2f}'.format(rmse_list[i]))
            print('{:.2f}%'.format(mape_net_list[i] * 100))
            print()
    else:
        return results


class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):

        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


def _get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        batch_size = kwargs['data'].get('batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        num_rnn_layers = kwargs['model'].get('num_rnn_layers')
        rnn_units = kwargs['model'].get('rnn_units')
        structure = '-'.join(['%d' % rnn_units for _ in range(num_rnn_layers)])

        # 保存的文件夹名称
        run_id = 'HIAM_%s_lr%g_bs%d_%s/' % (
            structure,
            learning_rate,
            batch_size,
            time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def init_weights(m):
    classname = m.__class__.__name__  # 2
    if classname.find('Conv') != -1 and classname.find('GAT') == -1:
        xavier_uniform_(m.weight.data)
    if type(m) == nn.Linear:
        xavier_uniform_(m.weight.data)
        # xavier_uniform_(m.bias.data)


from torch_geometric_temporal.nn.recurrent import DCRNN
import torch.nn.functional as F
import pickle

od_type="OD"
base_dir= f"../data/suzhou/"
prefix="train"

hyperparams_path = os.path.join(base_dir, f'{od_type.upper()}/hyperparameters.pkl')
with open(hyperparams_path, 'rb') as f:
    hyperparameters = pickle.load(f)

RGCN_node_features = hyperparameters['RGCN_node_features']
RGCN_hidden_units = hyperparameters['RGCN_hidden_units']
RGCN_output_dim = hyperparameters['RGCN_output_dim']
RGCN_K = hyperparameters['RGCN_K']

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_units, output_dim, K):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, hidden_units, K)
        self.linear = torch.nn.Linear(hidden_units, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.selu(h)
        h = self.linear(h)
        return h


def main(args):
    cfg = read_cfg_file(args.config_filename)
    log_dir = _get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    #  all edge_index in same dataset is same
    logger.info(cfg)
    batch_size = cfg['data']['batch_size']
    seq_len = cfg['model']['seq_len']
    horizon = cfg['model']['horizon']

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

    if not isinstance(graph_pkl_filename, list):
        graph_pkl_filename = [graph_pkl_filename]

    src = []
    dst = []
    for g in graph_pkl_filename:
        adj_mx = utils.load_graph_data(g)
        for i in range(len(adj_mx)):  # 构建邻接矩阵
            adj_mx[i, i] = 0
        adj_mx_list.append(adj_mx)

    adj_mx = np.stack(adj_mx_list, axis=-1)
    print("adj_mx:", adj_mx.shape)
    if cfg['model'].get('norm', False):
        print('row normalization')
        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)  # 归一化
    src, dst = adj_mx.sum(axis=-1).nonzero()
    print("src, dst:", src.shape, dst.shape)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_mx[adj_mx.sum(axis=-1) != 0],
                             dtype=torch.float,
                             device=device)
    print("train, edge:", edge_index.shape, edge_attr.shape)
    output_dim = cfg['model']['output_dim']
    for i in range(adj_mx.shape[-1]):
        logger.info(adj_mx[..., i])

    ## load dataset
    dataset = utils.load_dataset(**cfg['data'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset.items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    scaler_od = dataset['scaler']
    scaler_od_torch = utils.StandardScaler_Torch(scaler_od.mean,
                                                 scaler_od.std,
                                                 device=device)
    logger.info('scaler_od.mean:{}, scaler_od.std:{}'.format(scaler_od.mean,
                                                             scaler_od.std))

    # Define model with additional feature dimension
    model = Net_att(cfg, logger).to(device)
    model.apply(init_weights)
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['train']['base_lr'],
                           eps=cfg['train']['epsilon'])
    scheduler = StepLR2(optimizer=optimizer,
                        milestones=cfg['train']['steps'],
                        gamma=cfg['train']['lr_decay_ratio'],
                        min_lr=cfg['train']['min_learning_rate'])

    max_grad_norm = cfg['train']['max_grad_norm']
    train_patience = cfg['train']['patience']

    update = {}
    for category in ['od']:
        update['val_steady_count_' + category] = 0
        update['last_val_mae_' + category] = 1e6
        update['last_val_mape_net_' + category] = 1e6

    horizon = cfg['model']['horizon']
    import pickle
    with open('../data/suzhou/graph_sz_conn_no_11.pkl', 'rb') as f:
        graph_sz_conn_no_11 = pickle.load(f, errors='ignore')
    row, col = np.nonzero(graph_sz_conn_no_11)
    edge_index = np.array([row, col])
    edge_weight = graph_sz_conn_no_11[row, col]
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    for epoch in range(cfg['train']['epochs']):
        total_loss = 0
        i = 0
        begin_time = time.perf_counter()
        dataset['train_loader'].shuffle()
        train_iterator = dataset['train_loader'].get_iterator()
        model.train()
        for _, (
        x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
        PINN_od_additional_features, OD_feature_array,
        #OD_path_compressed_array,
        Time_DepartFreDic_Array, repeated_sparse_2D_tensors,
        repeated_sparse_3D_tensors,
        # repeated_sparse_4D_tensors, repeated_sparse_5D_tensors
        ) in enumerate(train_iterator):
            optimizer.zero_grad()
            y_od = y_od[:, :horizon, :, :output_dim]
            sequences, sequences_y, y_od = collate_wrapper(
                x_od=x_od, y_od=y_od,
                unfinished=unfinished, history=history, yesterday=yesterday,
                PINN_od_features=PINN_od_features,
                PINN_od_additional_features=PINN_od_additional_features,
                OD_feature_array=OD_feature_array,
                # OD_path_compressed_array =OD_path_compressed_array,
                Time_DepartFreDic_Array = Time_DepartFreDic_Array,
                repeated_sparse_2D_tensors = repeated_sparse_2D_tensors,
                repeated_sparse_3D_tensors=repeated_sparse_3D_tensors,
                # repeated_sparse_4D_tensors=repeated_sparse_4D_tensors,
                # repeated_sparse_5D_tensors=repeated_sparse_5D_tensors,
                edge_index=edge_index, edge_attr=edge_attr,
                device=device, seq_len=seq_len, horizon=horizon)  # generate batch data

            y_od_pred = model(sequences, sequences_y)

            # OD loss
            y_od_pred = scaler_od_torch.inverse_transform(y_od_pred)  # *std+mean
            y_od = scaler_od_torch.inverse_transform(y_od)
            loss_od = criterion(y_od_pred, y_od)

            cost_PINN = 0
            for i_sub_features in range(PINN_od_features.shape[0]):
                sub_PINN_od_features = PINN_od_features[i_sub_features]
                sub_od_additional_features = PINN_od_additional_features[i_sub_features]
                # PINN Loss
                zero_tensor = torch.zeros((PINN_od_features.shape[1], PINN_od_features.shape[2]), device=device)
                nested_list_with_arrays = [zero_tensor[i].cpu().numpy() for i in range(PINN_od_features.shape[1])]
                sub_PINN_od_features_list_of_arrays = [sub_PINN_od_features[i] for i in
                                                       range(sub_PINN_od_features.shape[0])]
                sub_od_additional_features_list_of_arrays = [sub_od_additional_features[i] for i in
                                                             range(sub_od_additional_features.shape[0])]
                signal_dict = {
                    'features': sub_PINN_od_features_list_of_arrays,
                    'targets': nested_list_with_arrays,
                    'additional_feature': sub_od_additional_features_list_of_arrays,
                    'edge_index': edge_index,
                    'edge_weight': edge_weight
                }
                from torch_geometric_temporal import StaticGraphTemporalSignal
                trip_gnr_signal = StaticGraphTemporalSignal(
                    features=signal_dict["features"],
                    targets=signal_dict["targets"],
                    additional_feature1=signal_dict["additional_feature"],
                    edge_index=signal_dict["edge_index"],
                    edge_weight=signal_dict["edge_weight"]
                )
                RecurrentGCN_trip_prdc = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units, output_dim=RGCN_output_dim,
                                      K=RGCN_K)

                od_type = "OD"
                base_dir = f"../data/suzhou/"

                RecurrentGCN_model_path = os.path.join(base_dir, f'{od_type.upper()}/RecurrentGCN_model.pth')
                RecurrentGCN_trip_prdc.load_state_dict(torch.load(RecurrentGCN_model_path))
                RecurrentGCN_trip_prdc.eval()
                y_od_pred_prdc = y_od_pred.sum(dim=-1)
                with torch.no_grad():
                    for snap_time, snapshot in enumerate(trip_gnr_signal):
                        y_hat = RecurrentGCN_trip_prdc(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                        y_hat = torch.tensor(y_hat, device=device)
                        y_od_pred_ = y_od_pred_prdc[i_sub_features][snap_time]
                        cost_PINN = cost_PINN + abs(torch.mean((y_hat - y_od_pred_)))

                RecurrentGCN_model_path = os.path.join(base_dir, f'{od_type.upper()}/RecurrentGCN_model.pth')
                RecurrentGCN_trip_attr = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units,
                                                      output_dim=RGCN_output_dim,
                                                      K=RGCN_K)
                RecurrentGCN_trip_attr.load_state_dict(torch.load(RecurrentGCN_model_path))
                RecurrentGCN_trip_attr.eval()
                y_od_pred_attr = y_od_pred.sum(dim=-2)
                with torch.no_grad():
                    for snap_time, snapshot in enumerate(trip_gnr_signal):
                        y_hat = RecurrentGCN_trip_attr(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                        y_hat = torch.tensor(y_hat, device=device)
                        y_od_pred_ = y_od_pred_attr[i_sub_features][snap_time]
                        cost_PINN = cost_PINN + abs(torch.mean((y_hat - y_od_pred_)))

            loss = loss_od + 0.001 * cost_PINN
            total_loss += loss.item()
            loss.backward()

            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            i += 1

        # evaluation on validation set
        val_result = evaluate(model=model,
                              dataset=dataset,
                              dataset_type='val',
                              edge_index=edge_index,
                              edge_attr=edge_attr,
                              device=device,
                              seq_Len=seq_len,
                              horizon=horizon,
                              output_dim=output_dim,
                              logger=logger,
                              detail=False,
                              cfg=cfg)

        time_elapsed = time.perf_counter() - begin_time

        logger.info(('Epoch:{}, total_loss:{}').format(epoch, total_loss / i))
        val_category = ['od']
        for category in val_category:
            logger.info('{}:'.format(category))
            logger.info(('val_mae:{}, val_mape_net:{}'
                         'r_loss={:.2f},lr={},  time_elapsed:{}').format(
                val_result['MAE_' + category],
                val_result['MAPE_net_' + category],
                0,
                str(scheduler.get_lr()),
                time_elapsed))
            if update['last_val_mae_' + category] > val_result['MAE_' + category]:
                logger.info('val_mae decreased from {} to {}'.format(
                    update['last_val_mae_' + category],
                    val_result['MAE_' + category]))
                update['last_val_mae_' + category] = val_result['MAE_' + category]
                update['val_steady_count_' + category] = 0
            else:
                update['val_steady_count_' + category] += 1

            if update['last_val_mape_net_' + category] > val_result['MAPE_net_' + category]:
                logger.info('val_mape_net decreased from {} to {}'.format(
                    update['last_val_mape_net_' + category],
                    val_result['MAPE_net_' + category]))
                update['last_val_mape_net_' + category] = val_result['MAPE_net_' + category]

        #  after per epoch, run evaluation on test dataset
        if (epoch + 1) % cfg['train']['test_every_n_epochs'] == 0:
            evaluate(model=model,
                     dataset=dataset,
                     dataset_type='test',
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     device=device,
                     seq_Len=seq_len,
                     horizon=horizon,
                     output_dim=output_dim,
                     logger=logger,
                     cfg=cfg)

        if (epoch + 1) % cfg['train']['save_every_n_epochs'] == 0:
            save_dir = log_dir
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            config_path = os.path.join(save_dir,
                                       'config-{}.yaml'.format(epoch + 1))
            epoch_path = os.path.join(save_dir,
                                      'epoch-{}.pt'.format(epoch + 1))
            torch.save(model.state_dict(), epoch_path)
            with open(config_path, 'w') as f:
                from copy import deepcopy
                save_cfg = deepcopy(cfg)
                save_cfg['model']['save_path'] = epoch_path
                f.write(yaml.dump(save_cfg, Dumper=Dumper))

        if train_patience <= update['val_steady_count_od']:
            logger.info('early stopping.')
            break
        scheduler.step()


if __name__ == "__main__":
    main(args)
