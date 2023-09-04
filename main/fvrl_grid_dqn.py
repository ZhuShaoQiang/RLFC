# -*- coding: utf-8 -*-

"""
网格世界版本
从经验中学习打分器，然后指导agent学习的算法

由于网格世界比较简单，可以用一个固定的key-value映射作为打分器
"""

import sys
import os
sys.path.append(os.getcwd())

# 导入参数
from config.grid_cliffwalking_dqn_ppo import params

def main():
    # 创建日志和保存的文件夹
    os.makedirs(params["LOGS_PATH"], exist_ok=True)
    os.makedirs(params["SAVE_PATH"], exist_ok=True)

    env = params["env"](step_reward=params["step_reward"], dead_reward=params["dead_reward"], goal_reward=params["goal_reward"],
                 total_col=params["total_col"], total_row=params["total_row"])

    # dqn = 

if __name__ == "__main__":
    main()
