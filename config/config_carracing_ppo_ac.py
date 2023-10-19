# dqn的配置
import sys
import os
sys.path.append(os.getcwd())
import torch
import gym
CarRacing = gym.make("CarRacing-v2")

params = {
    # 打分器的配置
    "scorer_activation": torch.nn.LeakyReLU(),
    "scorer_eps": 1,  # 打分器的系数
    "train_scorer_epoch": 2,  # 训练1000轮打分器

    "buffer_size": 2048,  # 回放缓冲区放置三十万个经验
    "train_total_episodes": 2_000,  # 总共训练的episode总数
    "train_num_epoch": 10,  # 训练1个epoch
    "batch_size": 64,
    "save_every": 100,

    "epsilon": 0.05,  # 贪心算法的探索率
    "epsilon_decay": True,
    "init_epsilon": 1.0,
    "min_epsilon": 0.05,
    "decay_ratio": 0.9975,
    "lr": 3e-4,
    "last_n_avg_rewards": 50,
    "seed": 42,

    "device": torch.device("cuda"),
    "activation": torch.nn.ReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,

    "env": CarRacing,
    "gamma": 0.99,
    "use_gae": True,  # 不使用gae算法
    "gae_lambda": 0.95,

    "LOGS_PATH": "./ckp/LOGS/PPOCarRacing/",
    "SAVE_PATH": "./ckp/OUT/PPOCarRacing/",
    "SCORER_PATH": "./ckp/OUT/PPOCarRacing/scorer.pth",  # 打分器是一样的，可以重用
}
