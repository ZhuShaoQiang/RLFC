# 自己算法（fvrl）的配置
import sys
import os
sys.path.append(os.getcwd())
import torch
from Lib.env import CliffWalking

params = {
    "train_total_episodes": 2000,  # 总共训练的episode总数
    "train_num_epoch": 10,  # 每个episode之后重复训练多少次
    "batch_size": 64,
    "buffer_size": 1001,  # 这个是最大值
    "epsilon": 0.05,  # 贪心算法的探索率
    "lr": 0.001,
    "last_n_avg_rewards": 50,
    "seed": 42,

    "device": torch.device("cuda"),
    "activation": torch.nn.LeakyReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,

    "env": CliffWalking,
    "gamma": 0.99,
    "step_reward": -0.1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 12,
    "total_row": 4,

    "LOGS_PATH": "./ckp/LOGS/cliffwalking_DQN005/",
    "SAVE_PATH": "./ckp/OUT/cliffwalking_DQN005/",
}
