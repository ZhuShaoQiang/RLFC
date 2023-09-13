# 自己算法（fvrl）的配置
import sys
import os
sys.path.append(os.getcwd())
import torch
from Lib.env import CliffWalking

params = {
    # 下面是训练经验打分器的部分
    "onehot": True,
    "train_scorer_epoch": 200,  # 打分器训练多少epoch
    "scorer_activation": torch.nn.LeakyReLU(),

    # 下面是训练RL算法的设置
    "train_total_episodes": 1000,  # 总共训练的episode总数
    "train_num_epoch": 10,  # 每个episode之后重复训练多少次
    "batch_size": 64,
    "buffer_size": 1001,  # 这个是最大值
    "epsilon": 0.05,  # 贪心算法的探索率，当不衰减的时候，走这个
    "lr": 0.001,
    "last_n_avg_rewards": 50,
    "seed": 42,
    "epsilon_decay": True,  # 是否衰减epsilon
    "min_epsilon": 0.05,  # 最小的探索率
    "init_epsilon": 1.0,  # 最初始的探索率，会覆盖掉上面的epsilon
    "decay_ratio": 0.99,  # 衰减率

    "device": torch.device("cuda"),
    "activation": torch.nn.LeakyReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "scorer_eps": 0,  # 用于乘给打分器作为一个系数
    # 这个分数为0的时候，应该就是普通的dqn

    "env": CliffWalking,
    "gamma": 0.99,
    "step_reward": -0.1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 12,
    "total_row": 4,

    "LOGS_PATH": "./ckp/LOGS/cliffwalking_DQN_rlfc_0_w_decay0.05/",
    "SAVE_PATH": "./ckp/OUT/cliffwalking_DQN_rlfc_0_w_decay0.05/",
    "SCORER_PATH": "./ckp/OUT/cliffwalking_DQN_rlfc/scorer.pth",
}
