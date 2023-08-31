# 自己算法（fvrl）的配置
from ..Lib.env import CliffWalking

params = {
    "lr": 0.001,
    "batch_size": 8,
    "train_total_episodes": 500,  # 总共训练的episode总数
    "train_num_epoch": 10,  # 每个episode之后重复训练多少次

    "env": CliffWalking,
    "step_reward": -0.1,
    "dead_reward": -10,
    "goal_reward": 10,
    "total_col": 12,
    "total_row": 4,


    "LOGS_PATH": "./ckp/LOGS/cliffwalking_DQN/",
    "SAVE_PATH": "./ckp/OUT/cliffwalking_DQN/",

}
