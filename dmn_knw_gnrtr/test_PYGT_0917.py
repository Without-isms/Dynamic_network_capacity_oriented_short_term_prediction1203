from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
import os


def test_PYGT(base_dir, prefix, od_type, str_prdc_attr):
    dir_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_test_dataset.pkl')
    with open(dir_path, 'rb') as f:
        test_dataset = pickle.load(f, errors='ignore')

    hyperparams_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}hyperparameters.pkl')
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

    RecurrentGCN_model = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units, output_dim=RGCN_output_dim,
                                      K=RGCN_K)

    model_path = os.path.join(base_dir, f"{od_type.upper()}{os.path.sep}{str_prdc_attr}_RecurrentGCN_model.pth")
    RecurrentGCN_model.load_state_dict(torch.load(model_path))

    RecurrentGCN_model.eval()
    cost_PINN = 0
    with torch.no_grad():
        for snap_time, snapshot in enumerate(test_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            cost_PINN = cost_PINN + mape
        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN = cost_PINN.item()

    print("MSE: {:.4f}".format(cost_PINN))


"""od_type="OD"
base_dir=f"data{os.path.sep}suzhou{os.path.sep}"
prefix="train"
str_prdc_attr=[]
test_PYGT(base_dir,prefix,od_type,str_prdc_attr)"""