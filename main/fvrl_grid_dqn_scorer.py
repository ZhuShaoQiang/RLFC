# -*- coding: utf-8 -*-

"""
网格世界版本
从经验中学习打分器，然后指导agent学习的算法

这个是打分器的
"""

import sys
import os
sys.path.append(os.getcwd())
import random

import numpy as np
import torch
from torch import nn

# 导入参数
from config.grid_cliffwalking_rlfc import params
from Lib.models import RLFCScorer
from Lib.utils import set_seed

from exp import initialize_exp, CLIFFWALKING_EXP

def main():
    """
    这个文件是为了训练打分器，不需要日志，不需要算法，只需要一个网络，一个经验（数据集）加载器即可
    """
    # 创建日志和保存的文件夹
    os.makedirs(params["LOGS_PATH"], exist_ok=True)
    os.makedirs(params["SAVE_PATH"], exist_ok=True)

    ### 1. 先初始化经验，得到经验的加载器
    train_exp_loader, test_exp_loader = initialize_exp(CLIFFWALKING_EXP, params)
    """
    for i in exp_loader:
        print(i)
    """

    ### 2. 然后使用一个前馈神经网络，预训练一个打分器
    model = RLFCScorer(
        input_dim=params["total_row"]*params["total_col"]*3, output_dim=1, activation=params["scorer_activation"],
        hidden=[32*3, 16*3], last_activation=nn.Sigmoid()
    ).to(params["device"])

    ### 3. 设定损失函数、优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params["lr"]
    )

    ### 4. 训练
    y = torch.tensor([[1]], dtype=torch.float).to(params["device"])  # 正例的目标
    y_ = torch.tensor([[0]], dtype=torch.float).to(params["device"])  # 负例的目标，为0
    for epoch in range(params["train_scorer_epoch"]):
        print(f"正在训练第{epoch}个epoch:", end="", flush=True)
        train_loss = 0
        test_loss = 0
        for exp in train_exp_loader:  # 拿到经验，s是一整条轨迹
            # 每一个i是一个列表，里面是 [s1, s2, s3, s4]
            for idx in range(1, len(exp)-1):  # 下标为  1 2
                s1 = exp[idx-1]
                s2 = exp[idx]
                s3 = exp[idx+1]
                # 现在的逻辑是，通过3个状态，做成一个时序型判断，s1看作前一时刻，s2看作当前，s3看作后一时刻
                # 只要s1 s2之后是s3，就应该是高分，否则低分（0）

                s_random = torch.zeros_like(s1)
                random_idx = torch.randint(0, len(s1), ())
                s_random[random_idx] = 1  # 随机产生一个onehot，作为反例

                # 正例训练
                x = torch.cat([s1, s2, s3]).unsqueeze(0)
                pred_y = model.forward(x)
                optimizer.zero_grad()
                loss = loss_fn(pred_y, y)
                loss.backward()
                optimizer.step()
                train_loss += loss

                s_random = random.choice([s1, s2, s_random])  # 要么随机，要么回走，要么原地踏步
                # 反例1训练，反例有两种，一种是走反了，一种是原地踏步，原地踏步应该也是-1分
                x_ = torch.cat([s1, s2, s_random]).unsqueeze(0)
                pred_y_ = model.forward(x_)
                optimizer.zero_grad()
                loss = loss_fn(pred_y_, y_)
                loss.backward()
                optimizer.step()
                train_loss += loss

        # 每训练一个epoch，就测试，测试时，应有随机测试，和反向走测试，和原地踏步测试
        with torch.no_grad():
            for exp in test_exp_loader:  # 拿到经验，s是一整条轨迹
                # 每一个i是一个列表，里面是 [s1, s2, s3, s4]
                for idx in range(len(exp)-1):
                    s1 = exp[idx-1]
                    s2 = exp[idx]
                    s3 = exp[idx+1]
                    # 现在的逻辑是，通过3个状态，做成一个时序型判断，s1看作前一时刻，s2看作当前，s3看作后一时刻
                    # 只要s1 s2之后是s3，就应该是高分，否则低分（0）

                    s_random = torch.zeros_like(s1)
                    random_idx = torch.randint(0, len(s1), ())
                    s_random[random_idx] = 1  # 随机产生一个onehot，作为反例

                    # 正例测试
                    x = torch.cat([s1, s2, s3]).unsqueeze(0)
                    test_pred_y = model.forward(x)
                    loss = loss_fn(test_pred_y, y)
                    test_loss += loss

                    # 反例测试，后面随机状态
                    x_ = torch.cat([s1, s2, s_random]).unsqueeze(0)
                    pred_y_ = model.forward(x_)
                    loss = loss_fn(pred_y_, y_)
                    test_loss += loss
                    # 反例测试，后面原地踏步
                    x_ = torch.cat([s1, s2, s2]).unsqueeze(0)
                    pred_y_ = model.forward(x_)
                    loss = loss_fn(pred_y_, y_)
                    test_loss += loss
                    # 反例测试，后面掉头回去
                    x_ = torch.cat([s1, s2, s1]).unsqueeze(0)
                    pred_y_ = model.forward(x_)
                    loss = loss_fn(pred_y_, y_)
                    test_loss += loss

        print(f"Train Loss: {train_loss:.4f}, eval loss: {test_loss:.4f}")
    torch.save(model.state_dict(),
               params["SCORER_PATH"])

if __name__ == "__main__":
    set_seed(params["seed"])
    main()
