# 自己算法（fvrl）的配置，运行halfcheetah游戏
import sys
import os
sys.path.append(os.getcwd())
import torch
import gymnasium

params = {
    # 下面是训练经验打分器的部分
    "train_scorer_epoch": 2,  # 打分器训练多少epoch
    "scorer_activation": torch.nn.Sigmoid,
    "time_len": 3,  # 时间长度，t-n+1到t+1，长度为n
    "split_ratio_exp": 0.2,  # 使用多少经验，20万条

    # 下面是训练RL算法的设置
    "env_name": "CliffWalking-v0",
    "train_total_episodes": 1000,  # 总共训练的episode总数
    "train_num_epoch": 40,  # 每个episode之后重复训练多少次
    "batch_size": 64,
    "buffer_size": 2048,  # 这个是最大值
    "epsilon": 0.05,  # 贪心算法的探索率，当不衰减的时候，走这个
    "lr": 3e-4,
    "last_n_avg_rewards": 10,
    "seed": 42,
    "epsilon_decay": True,  # 是否衰减epsilon
    "min_epsilon": 0.00,  # 最小的探索率
    "init_epsilon": 0.2,  # 最初始的探索率，会覆盖掉上面的epsilon
    "decay_ratio": 0.99,  # 衰减率
    "save_every": 100,

    "device": "cuda:0",
    "activation": torch.nn.LeakyReLU(),
    "optimizer": torch.optim.Adam,
    "clip_range": 0.2,
    "entropy_coef": 0.0,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_gae": False,
    "gae_lambda": 0.0,

    "gamma": 0.99,
    "step_reward": -0.1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 12,
    "total_row": 4,

    "SCORER_PATH": "./ckp/pth/cw_dqn_scorer.pth",
}
### 打分器的分数降低
params["scorer_eps"] = 0.20  # 打分器权重占比，占比为0就算普通的算法
# 打分器权重是否衰减
params["scorer_eps_decay"] = True
# 打分器初始权重和权重一样
params["scorer_eps_init"] = params["scorer_eps"]
# 打分器最低权重
params["scorer_eps_min"] = 0.0
# 第200epoch就要衰减到0.0了
params["scorer_eps_end"] = 200
