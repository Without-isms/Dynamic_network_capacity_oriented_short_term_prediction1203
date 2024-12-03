# encoding:utf-8
import random
import argparse
import time
import yaml
import numpy as np
import torch
import os

from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_
from lib import utils_HIAM_button as utils
from lib.utils_HIAM_button import collate_wrapper
from lib import metrics
from models.Net_1004 import Net_1004
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--config_filename',
                    default=None,
                    type=str,
                    help='Configuration filename for restoring the model.')
args = argparse.Namespace(config_filename='data/config/eval_sz_dim26_units96_h4c512.yaml')


def read_cfg_file(filename):
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg

def run_model(model, data_iterator, edge_index, edge_attr, device, seq_len, horizon, output_dim,
              ENABLE_2D_3D_4D_COMPRESSED_FEATURES, ENABLE_5D_FEATURES):
    """
    return a list of (horizon_i, batch_size, num_nodes, output_dim)
    """
    # while evaluation, we need model.eval and torch.no_grad
    model.eval()
    y_od_pred_list = []
    for _, batch in enumerate(data_iterator):
        if ENABLE_2D_3D_4D_COMPRESSED_FEATURES and ENABLE_5D_FEATURES:
            (x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
             PINN_od_additional_features, OD_feature_array,
             Time_DepartFreDic_Array,
             # OD_path_compressed_array,
             repeated_sparse_2D_tensors, repeated_sparse_3D_tensors, repeated_sparse_4D_tensors,
             repeated_sparse_5D_tensors) = batch
        elif ENABLE_2D_3D_4D_COMPRESSED_FEATURES:
            (x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
             PINN_od_additional_features, OD_feature_array,
             Time_DepartFreDic_Array,
             # OD_path_compressed_array,
             repeated_sparse_2D_tensors, repeated_sparse_3D_tensors, repeated_sparse_4D_tensors) = batch
        elif ENABLE_5D_FEATURES:
            (x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
             PINN_od_additional_features, OD_feature_array,
             Time_DepartFreDic_Array, repeated_sparse_5D_tensors) = batch
        else:
            (x_od, y_od, unfinished, history, yesterday, xtime, ytime, PINN_od_features,
             PINN_od_additional_features, OD_feature_array,
             Time_DepartFreDic_Array) = batch
        y_od = y_od[..., :output_dim]
        sequences, sequences_y, y_od = collate_wrapper(
            x_od=x_od, y_od=y_od, unfinished=unfinished, history=history,
            yesterday=yesterday, PINN_od_features=PINN_od_features,
            PINN_od_additional_features=PINN_od_additional_features, OD_feature_array=OD_feature_array,
            Time_DepartFreDic_Array=Time_DepartFreDic_Array,
            edge_index=edge_index, edge_attr=edge_attr, device=device, seq_len=seq_len, horizon=horizon,
            #OD_path_compressed_array=OD_path_compressed_array if ENABLE_2D_3D_4D_COMPRESSED_FEATURES else None,
            repeated_sparse_2D_tensors=repeated_sparse_2D_tensors if ENABLE_2D_3D_4D_COMPRESSED_FEATURES else None,
            repeated_sparse_3D_tensors=repeated_sparse_3D_tensors if ENABLE_2D_3D_4D_COMPRESSED_FEATURES else None,
            repeated_sparse_4D_tensors=repeated_sparse_4D_tensors if ENABLE_2D_3D_4D_COMPRESSED_FEATURES else None,
            repeated_sparse_5D_tensors=repeated_sparse_5D_tensors if ENABLE_5D_FEATURES else None
        )
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
                ENABLE_2D_3D_4D_COMPRESSED_FEATURES,
                ENABLE_5D_FEATURES,
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
        output_dim=output_dim,
        ENABLE_2D_3D_4D_COMPRESSED_FEATURES=ENABLE_2D_3D_4D_COMPRESSED_FEATURES,
        ENABLE_5D_FEATURES = ENABLE_5D_FEATURES
    )

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
        # horizon = dataset['y_{}'.format(dataset_type)].shape[1]
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
        # return mae_sum / horizon, rmse_sum / horizon, mape_sta_sum / horizon, mape_pair_sum / horizon, mape_net_sum/ horizon, mape_distribution_sum / horizon
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

        # name of dir for saving log
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
    if classname.find('Conv') != -1 and classname.find('RGCN') == -1:
        xavier_uniform_(m.weight.data)
    if type(m) == nn.Linear:
        xavier_uniform_(m.weight.data)
        #xavier_uniform_(m.bias.data)

def toDevice(datalist, device):
    for i in range(len(datalist)):
        datalist[i] = datalist[i].to(device)
    return datalist

def main(args):
    cfg = read_cfg_file(args.config_filename)
    log_dir = _get_log_dir(cfg)
    log_level = cfg.get('log_level', 'INFO')

    logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    #  all edge_index in same dataset is same
    # edge_index = adjacency_to_edge_index(adj_mx)  # alreay added self-loop
    logger.info(cfg)
    batch_size = cfg['data']['batch_size']
    seq_len = cfg['model']['seq_len']
    horizon = cfg['model']['horizon']
    ENABLE_2D_3D_4D_COMPRESSED_FEATURES = cfg['domain_knowledge']['ENABLE_2D_3D_4D_COMPRESSED_FEATURES']
    ENABLE_5D_FEATURES = cfg['domain_knowledge']['ENABLE_5D_FEATURES']
    Using_GAT_or_RGCN = cfg['domain_knowledge']['Using_GAT_or_RGCN']
    # edge_index = utils.load_pickle(cfg['data']['edge_index_pkl_filename'])

    adj_mx_list = []
    graph_pkl_filename = cfg['data']['graph_pkl_filename']

    if not isinstance(graph_pkl_filename, list):
        graph_pkl_filename = [graph_pkl_filename]

    src = []
    dst = []
    for g in graph_pkl_filename:
        adj_mx = utils.load_graph_data(g)
        for i in range(len(adj_mx)):
            adj_mx[i, i] = 0
        adj_mx_list.append(adj_mx)

    adj_mx = np.stack(adj_mx_list, axis=-1)
    print("adj_mx:", adj_mx.shape)
    if cfg['model'].get('norm', False):
        print('row normalization')
        adj_mx = adj_mx / (adj_mx.sum(axis=0) + 1e-18)  
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

    dataset = utils.load_dataset(**cfg['data'], scaler_axis=(0, 1, 2, 3))
    for k, v in dataset.items():
        if hasattr(v, 'shape'):
            logger.info((k, v.shape))

    model = Net_1004(cfg, logger).to(device)
    state = torch.load(cfg['model']['save_path'])
    model.load_state_dict(state, strict=False)
    evaluate(model=model,
             dataset=dataset,
             dataset_type='test',
             edge_index=edge_index,
             edge_attr=edge_attr,
             device=device,
             seq_Len=seq_len,
             horizon=horizon,
             output_dim=output_dim,
             ENABLE_2D_3D_4D_COMPRESSED_FEATURES=ENABLE_2D_3D_4D_COMPRESSED_FEATURES,
             ENABLE_5D_FEATURES=ENABLE_5D_FEATURES,
             logger=logger,
             cfg=cfg)

if __name__ == "__main__":
    main(args)
