# -*- coding: utf-8 -*-

"""
使用PPO(AC)跑普通的CarRacing程序
"""

import sys
import os
sys.path.append(os.getcwd())

import torch
from torchsummary import summary

from Lib.algorithms import VanillaPPO, PPO_RLFC
from Lib.models import AC, CnnAC, RLFCScorer
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
    ### 加载模型
    scorer = RLFCScorer(
        input_dim=(4, 96, 96), output_dim=1,
        hidden=[
            ["conv2d", {
                "in_channels": 4,
                "out_channels": 32,
                "kernel_size": 8,
                "stride": 4,
            }],
            ["activation", {}],
            ["conv2d", {
                "in_channels": 32,
                "out_channels": 64,
                "kernel_size": 4,
                "stride": 2,
            }],
            ["activation", {}],
            ["conv2d", {
                "in_channels": 64,
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
            }],
            ["activation", {}],
            ["flatten", {}],
        1024, ["activation", {}]],
        last_activation=torch.nn.Sigmoid,
    ).to(params["device"])
    summary(ac, input_size=(3, 96, 96))

    # 使用PPO算法，传入这个模型
    model = PPO_RLFC(
        env=env, params=params, policy=ac, scorer=scorer
    )
    model.learn()
    model.save("fvrl_final.pth")


if __name__ == "__main__":
    set_seed(params["seed"])
    main()
