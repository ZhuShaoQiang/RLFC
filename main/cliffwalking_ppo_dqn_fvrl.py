# -*- coding: utf-8 -*-

"""
网格世界版本
带经验的普通的PPO算法（使用DQN网络）学习走cliffwalking的代码
"""

import sys
import os
sys.path.append(os.getcwd())
import torch
from torch import nn

from Lib.algorithms import PPO_RLFC
from Lib.models import RLFCScorer, DQN
from Lib.envwrappers import ToTensorWrapper
from Lib.utils import set_seed

# 导入参数
from config.grid_cliffwalking_dqn_ppo_rlfc import params

def main():
    # 创建日志和保存的文件夹
    os.makedirs(params["LOGS_PATH"], exist_ok=True)
    os.makedirs(params["SAVE_PATH"], exist_ok=True)

    # 创建一个环境
    env = params["env"](step_reward=params["step_reward"], dead_reward=params["dead_reward"], goal_reward=params["goal_reward"], total_col=params["total_col"], total_row=params["total_row"])
    
    env = ToTensorWrapper(env=env)
    # env.reset(seed=params["seed"])  # 如果使用gym的环境，这句话可以设定随机种子，但是我们这个环境不涉及随机，不需要设置

    # 加载普通的Dqn网络
    policy = DQN(
        input_dim=env.obs_space, output_dim=4, activation=params["activation"],
        hidden=[32, 16], last_activation=None
    )

    # 加载经验打分器，由于打分器自己有一个正交标准化，所以必须先加载DQN，防止DQN的正交标准化和普通的DQN的正交标准化不一样
    scorer = RLFCScorer(
        input_dim=params["total_row"]*params["total_col"]*3, output_dim=1, activation=params["scorer_activation"],
        hidden=[32*3, 16*3], last_activation=nn.Sigmoid()
    ).to(params["device"])
    scorer.load_state_dict(
        torch.load(params["SCORER_PATH"]), strict=True
    )  # 加载预训练的模型
    
    # 加载算法
    model = PPO_RLFC(env=env, policy=policy, scorer=scorer, params=params)
    model.learn()


if __name__ == "__main__":
    set_seed(params["seed"])
    main()
