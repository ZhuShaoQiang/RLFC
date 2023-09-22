# dqn的配置
import sys
import os
sys.path.append(os.getcwd())
import torch
from Lib.env import CliffWalking

params = {
    # 打分器的配置
    "scorer_activation": torch.nn.LeakyReLU(),
    "scorer_eps": 1,  # 打分器的系数

    "buffer_size": 300_000,  # 回放缓冲区放置三十万个经验
    "total_timesteps": 10_000,  # 总共训练的episode总数
    "learning_starts": 100,  # 100个steps之前不学习
    "train_freq": 4,  # 每走四步训练一次
    "train_num_epochs": 1,  # 训练1个epoch
    "batch_size": 64,

    "epsilon": 0.05,  # 贪心算法的探索率
    "epsilon_decay": True,
    "init_epsilon": 1.0,
    "min_epsilon": 0.05,
    "decay_ratio": 0.9975,
    "lr": 1e-3,
    "last_n_avg_rewards": 50,
    "seed": 42,

    "device": torch.device("cuda"),
    "activation": torch.nn.ReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 10,

    "env": CliffWalking,
    "gamma": 0.9,
    "step_reward": -1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 12,
    "total_row": 4,

    "LOGS_PATH": "./ckp/LOGS/DQNcliffwalking_005_rlfc/",
    "SAVE_PATH": "./ckp/OUT/DQNcliffwalking_005_rlfc/",
    "SCORER_PATH": "./ckp/OUT/cliffwalking_DQN_rlfc/scorer.pth",  # 打分器是一样的，可以重用
}
