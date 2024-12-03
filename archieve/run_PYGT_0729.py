from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import os

def run_PYGT(base_dir, prefix, od_type):
    dir_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}{prefix}_{od_type.upper()}_signal_dict.pkl')
    with open(dir_path, 'rb') as f:
        signal_dict = pickle.load(f, errors='ignore')
    # 实例化 StaticGraphTemporalSignal 对象
    signal = StaticGraphTemporalSignal(
        features=signal_dict["features"],
        targets=signal_dict["targets"],
        additional_feature1=signal_dict["additional_feature"],
        edge_index=signal_dict["edge_index"],
        edge_weight=signal_dict["edge_weight"]
    )

    from torch_geometric_temporal.signal import temporal_signal_split
    train_dataset, test_dataset = temporal_signal_split(signal, train_ratio=0.9)

    with open(os.path.join(base_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)

    import torch
    import torch.nn.functional as F
    from torch_geometric_temporal.nn.recurrent import DCRNN

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features):
            super(RecurrentGCN, self).__init__()
            self.recurrent = DCRNN(node_features, 6, 2)
            # self.dropout = torch.nn.Dropout(p=0.5)  # Dropout with 50% drop probability
            self.linear = torch.nn.Linear(6, 1)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.selu(h)
            # h = self.dropout(h)  # Apply Dropout
            h = self.linear(h)
            return h

    from tqdm import tqdm

    RecurrentGCN_model = RecurrentGCN(node_features=156)

    optimizer = torch.optim.Adam(RecurrentGCN_model.parameters(), lr=0.01)

    RecurrentGCN_model.train()
    epoch_num = 100
    for epoch in tqdm(range(2)):
        cost_PINN = 0
        total_mape = 0
        for snap_time, snapshot in enumerate(train_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost_PINN = cost_PINN + torch.mean((y_hat - snapshot.y) ** 2)
            # 计算 MAPE
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            total_mape += mape

        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_mape = total_mape / (snap_time + 1)

        print(f"Epoch {epoch + 1}/{epoch_num}, MSE: {cost_PINN.item():.4f}, MAPE: {avg_mape.item():.4f}%")

    model_save_path = os.path.join(base_dir, f'{od_type.upper()}{os.path.sep}RecurrentGCN_model.pth')
    torch.save(RecurrentGCN_model.state_dict(), model_save_path)
    print(f"模型已保存到 {model_save_path}")

    """
    RecurrentGCN_model.eval()
    cost_PINN = 0
    epsilon = 1e-8  # 添加一个很小的常数来避免除零

    # 加载模型
    RecurrentGCN_model.load_state_dict(torch.load("RecurrentGCN_model.pth"))


    RecurrentGCN_model.eval()
    cost_PINN = 0
    with torch.no_grad():
        for snap_time, snapshot in enumerate(test_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            cost_PINN = cost_PINN + mape
        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN = cost_PINN.item()

    print("MSE: {:.4f}".format(cost_PINN))
    """

"""
od_type="OD"
base_dir=f"C:{os.path.sep}Users{os.path.sep}Administrator{os.path.sep}PycharmProjects{os.path.sep}HIAM-main{os.path.sep}data{os.path.sep}suzhou{os.path.sep}{od_type.upper()}"
prefix="train_prdc"
dir_path = os.path.join(base_dir, f'{prefix}_{od_type.upper()}_signal_dict.pkl')
run_PYGT(dir_path)
"""