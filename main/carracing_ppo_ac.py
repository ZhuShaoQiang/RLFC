# -*- coding: utf-8 -*-

"""
使用PPO(AC)跑普通的CarRacing程序
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
from torchsummary import summary

from Lib.algorithms import VanillaPPO
from Lib.models import AC, CnnAC
from Lib.envwrappers import ImgToTensor
from Lib.utils import set_seed, compare_networks

from config.config_carracing_ppo_ac import params

def main():
    """
    """
    os.makedirs(params["SAVE_PATH"], exist_ok=True)
    os.makedirs(params["LOGS_PATH"], exist_ok=True)

    # 这个地方拿到环境
    env = params["env"]
    env.reset(seed=params["seed"])
    env = ImgToTensor(env, 96, 96)

    # 设定网络
    ac = CnnAC(
        state_dim=[3, 96, 96], action_dim=3, feature_dim=1024
    ).to("cuda")
    summary(ac, input_size=(3, 96, 96))

    # 使用PPO算法，传入这个模型
    model = VanillaPPO(
        env=env, params=params, policy=ac
    )
    model.learn()
    model.save("final.pth")


if __name__ == "__main__":
    set_seed(params["seed"])
    main()
