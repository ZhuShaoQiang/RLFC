# 自己算法（fvrl）的配置
import sys
import os
sys.path.append(os.getcwd())
import torch
from Lib.env import Adventure

params = {
    # 下面是训练经验打分器的部分
    "onehot": True,
    "train_scorer_epoch": 100,  # 打分器训练多少epoch
    "scorer_activation": torch.nn.LeakyReLU(),

    # 下面是训练RL算法的设置
    "total_timesteps": 10_00,  # 总共训练的episode总数
    "batch_size": 64,
    "buffer_size": 30_0000,  # 这个是最大值
    "learning_starts": 100,  # 100个steps之前不学习
    "epsilon": 0.05,  # 贪心算法的探索率，当不衰减的时候，走这个
    "lr": 1e-3,
    "last_n_avg_rewards": 50,
    "seed": 42,
    "epsilon_decay": True,  # 是否衰减epsilon
    "min_epsilon": 0.05,  # 最小的探索率
    "init_epsilon": 1.0,  # 最初始的探索率，会覆盖掉上面的epsilon
    "decay_ratio": 0.9975,  # 衰减率
    "train_freq": 4,  # 每走四步训练一次
    "train_num_epochs": 1,  # 训练1个epoch

    "device": torch.device("cuda"),
    "activation": torch.nn.LeakyReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 10,
    "scorer_eps": 0,  # 用于乘给打分器作为一个系数
    # 这个分数为0的时候，应该就是普通的dqn

    "env": Adventure,
    "gamma": 0.99,
    "step_reward": -1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 7,
    "total_row": 7,

    "LOGS_PATH": "./ckp/LOGS/DQNAdventure_DQN0.05/",
    "SAVE_PATH": "./ckp/OUT/DQNAdventure_DQN0.05/",
    "SCORER_PATH": "./ckp/OUT/Adventure_DQN0.05/scorer.pth",
}
