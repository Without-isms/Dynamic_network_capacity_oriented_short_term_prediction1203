from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
import os

def test_PYGT(base_dir, prefix, od_type):
    dir_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_test_dataset.pkl')
    with open(dir_path, 'rb') as f:
        test_dataset = pickle.load(f, errors='ignore')

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features):
            super(RecurrentGCN, self).__init__()
            self.recurrent = DCRNN(node_features, 6, 2)
            # self.dropout = torch.nn.Dropout(p=0.5)  # Dropout with 50% drop probability
            self.linear = torch.nn.Linear(6, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.selu(h)
            h = self.linear(h)
            return h

    from tqdm import tqdm

    RecurrentGCN_model = RecurrentGCN(node_features=156)

    optimizer = torch.optim.Adam(RecurrentGCN_model.parameters(), lr=0.01)

    """RecurrentGCN_model.train()

    for epoch in tqdm(range(100)):
        cost_PINN = 0
        for snap_time, snapshot in enumerate(train_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost_PINN = cost_PINN + torch.mean((y_hat-snapshot.y)**2)
        cost_PINN = cost_PINN / (snap_time+1)
        cost_PINN.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_save_path = "RecurrentGCN_model.pth"
    torch.save(RecurrentGCN_model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")"""
    # 加载模型
    RecurrentGCN_model.load_state_dict(torch.load(os.path.join(base_dir, f"{od_type.upper()}{os.path.sep}RecurrentGCN_model.pth")))

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

