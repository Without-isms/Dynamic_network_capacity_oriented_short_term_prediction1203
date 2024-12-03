import numpy as np
from scipy.optimize import minimize
import os
from metro_data_convertor.Find_project_root import Find_project_root
import pickle
import torch
from torch.optim import Adam

"""def impedance_function(C, gamma):
    return np.power(C, -gamma)"""

def impedance_function(C, gamma):
    C = torch.tensor(C, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    return torch.pow(C, -gamma)

"""def compute_flow(O, D, C, gamma, a, b):
    f_c = impedance_function(C, gamma)
    q_v = np.outer(a, b) * O[:, np.newaxis].flatten() * D[np.newaxis, :].flatten() * f_c
    np.fill_diagonal(q_v, 0) 
    return q_v"""

def compute_flow(O, D, C, gamma, a, b):
    f_c = impedance_function(C, gamma)

    if isinstance(a, torch.Tensor) == False:
        a = torch.tensor(a, dtype=torch.float32)
    if isinstance(b, torch.Tensor) == False:
        b = torch.tensor(b, dtype=torch.float32)
    if isinstance(O, torch.Tensor) == False:
        O = torch.tensor(O, dtype=torch.float32)
    if isinstance(D, torch.Tensor) == False:
        D = torch.tensor(D, dtype=torch.float32)
    if isinstance(f_c, torch.Tensor) == False:
        f_c = torch.tensor(f_c, dtype=torch.float32)

    q_v = torch.ger(a, b) * O[:, None].flatten() * D[None, :].flatten() * f_c
    q_v.fill_diagonal_(0)
    return q_v

def objective_function(params, O_data, D_data, C_data, q_obs_data, time_steps):
    gamma = params[0]
    a = params[1:len(O_data[0]) + 1]
    b = params[len(O_data[0]) + 1:]

    total_mse = 0
    for t in range(time_steps):
        O = O_data[t]
        D = D_data[t]
        C = C_data
        q_obs = q_obs_data[t]

        q_v = compute_flow(O, D, C, gamma, a, b)

        mse = np.mean((q_v - q_obs) ** 2)
        total_mse += mse

    return total_mse


'''def fit_flow_model(O_data, D_data, C_data, q_obs_data, time_steps, initial_gamma=0.5, tol=1e-2, maxiter=1):
    """
    参数：
    O_data: list of np.array，出发量数据，按时间点存储。
    D_data: list of np.array，到达量数据，按时间点存储。
    C_data: np.array，交通阻抗矩阵，假设各时间点的阻抗矩阵相同。
    q_obs_data: list of np.array，观测到的流量数据，按时间点存储。
    time_steps: int，时间步数。
    initial_gamma: float，初始 gamma 参数，默认值为 0.5。
    tol: float，优化容差，默认值为 1e-2。
    maxiter: int，最大迭代次数，默认值为 1000。

    返回：
    optimal_gamma: float，最优的 gamma 参数。
    a_fitted: np.array，最优的 a 参数。
    b_fitted: np.array，最优的 b 参数。
    q_predicted_list: list of np.array，每个时间点的预测流量矩阵。


    O_data = [
    np.array([100, 150, 200]),  # 时间点1的出发量 O
    np.array([110, 160, 190]),  # 时间点2的出发量 O
    np.array([120, 170, 180])   # 时间点3的出发量 O
    ]

    D_data = [
        np.array([120, 180, 150]),  # 时间点1的到达量 D
        np.array([130, 190, 160]),  # 时间点2的到达量 D
        np.array([140, 200, 170])   # 时间点3的到达量 D
    ]

    C_data = [
        np.array([[0, 20, 30], [15, 0, 35], [20, 30, 0]]),  # 时间点1的交通阻抗矩阵 C
    ]

    q_obs_data = [
        np.array([[30, 40, 30], [40, 60, 50], [50, 80, 70]]),  # 时间点1的观测流量 q_obs
        np.array([[32, 42, 32], [42, 62, 52], [52, 82, 72]]),  # 时间点2的观测流量 q_obs
        np.array([[34, 44, 34], [44, 64, 54], [54, 84, 74]])   # 时间点3的观测流量 q_obs
    ]
    """

    # 初始化 a 和 b 参数
    initial_a = np.ones(len(O_data[0]))  # 初始化 a_i
    initial_b = np.ones(len(D_data[0]))  # 初始化 b_j

    # 拼接初始参数
    initial_params = np.concatenate(([initial_gamma], initial_a, initial_b))

    # 优化参数
    result = minimize(objective_function, initial_params, args=(O_data, D_data, C_data, q_obs_data, time_steps),
                      method='BFGS', options={'maxiter': maxiter, 'disp': True}, tol=tol)

    # 最优的 gamma、a 和 b 值
    optimal_params = result.x
    optimal_gamma = optimal_params[0]
    a_fitted = optimal_params[1:len(O_data[0]) + 1]
    b_fitted = optimal_params[len(O_data[0]) + 1:]

    # 预测每个时间点的流量矩阵，并保存结果
    q_predicted_list = []
    for t in range(time_steps):
        O_new = O_data[t]
        D_new = D_data[t]
        q_predicted = compute_flow(O_new, D_new, C_data, optimal_gamma, a_fitted, b_fitted)
        q_predicted_list.append(q_predicted)
        # 打印结果
        print(f'最优的 gamma 参数: {optimal_gamma}')
        print(f'最优的 a 参数: {a_fitted}')
        print(f'最优的 b 参数: {b_fitted}')
        for t, q_pred in enumerate(q_predicted_list):
            print(f"时间点 {t + 1} 的拟合流量矩阵 q_v:")
            print(q_pred)
    return optimal_gamma, a_fitted, b_fitted, q_predicted_list'''

def fit_trip_generation_model(O_data, D_data, C_data, q_obs_data, time_steps, initial_gamma, lr_gener, maxiter):
    """
    Parameters:
    O_data: list of torch.Tensor, departure volume data stored by time points.
    D_data: list of torch.Tensor, arrival volume data stored by time points.
    C_data: torch.Tensor, transportation impedance matrix, assuming the impedance matrix is the same at each time point.
    q_obs_data: list of torch.Tensor, observed flow data stored by time points.
    time_steps: int, number of time steps.
    initial_gamma: float, initial gamma parameter, default value is 0.5.
    lr_gener: float, learning rate.
    maxiter: int, maximum number of iterations, default value is 1000.

    Returns:
    optimal_gamma: float, optimal gamma parameter.
    a_fitted: torch.Tensor, optimal a parameter.
    b_fitted: torch.Tensor, optimal b parameter.
    q_predicted_list: list of torch.Tensor, predicted flow matrices for each time point.
    """

    initial_a = torch.ones(len(O_data[0]), requires_grad=True)
    initial_b = torch.ones(len(D_data[0]), requires_grad=True)
    gamma = torch.tensor([initial_gamma], requires_grad=True)

    optimizer = Adam([gamma, initial_a, initial_b], lr=lr_gener)

    for i in range(maxiter):
        optimizer.zero_grad()

        loss = objective_function(torch.cat([gamma, initial_a, initial_b]), O_data, D_data, C_data, q_obs_data,
                                  time_steps)

        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    optimal_gamma = gamma.item()
    a_fitted = initial_a.detach()
    b_fitted = initial_b.detach()

    q_predicted_list = []
    for t in range(time_steps):
        O_new = O_data[t]
        D_new = D_data[t]
        q_predicted = compute_flow(O_new, D_new, C_data, optimal_gamma, a_fitted, b_fitted)
        q_predicted_list.append(q_predicted)

        print(f'Optimal gamma parameter: {optimal_gamma}')
        print(f'Optimal a parameter: {a_fitted}')
        print(f'Optimal b parameter: {b_fitted}')
        for t, q_pred in enumerate(q_predicted_list):
            print(f"Fitted flow matrix q_v at time point {t + 1}:")
            print(q_pred)

    return optimal_gamma, a_fitted, b_fitted, q_predicted_list
