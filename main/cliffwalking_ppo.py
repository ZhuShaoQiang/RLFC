# -*- coding: utf-8 -*-

"""
网格世界版本
普通的PPO算法（使用DQN网络）学习走cliffwalking的代码
"""

import sys
import os
sys.path.append(os.getcwd())

# 导入参数
from config.fvrl_grid_ppo import params

def main():
    # 创建日志和保存的文件夹
    os.makedirs(params["LOGS_PATH"], exist_ok=True)
    os.makedirs(params["SAVE_PATH"], exist_ok=True)

    # 创建一个环境
    env = params["env"](step_reward=params["step_reward"], dead_reward=params["dead_reward"], goal_reward=params["goal_reward"],
                 total_col=params["total_col"], total_row=params["total_row"])

    # dqn = 

if __name__ == "__main__":
    main()
