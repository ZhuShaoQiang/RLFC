# -*- coding: utf-8 -*-

"""
网格世界版本
从经验中学习打分器，然后指导agent学习的算法

这个是打分器的
"""

import sys
import os
sys.path.append(os.getcwd())

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
        input_dim=params["total_row"]*params["total_col"]*2, output_dim=1, activation=params["scorer_activation"],
        hidden=[64, 32], last_activation=nn.Sigmoid()
    ).to(params["device"])

    ### 3. 设定损失函数、优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=params["lr"]
    )

    ### 4. 训练
    y = torch.tensor([[1]], dtype=torch.float).to(params["device"])  # 正例的目标
    y_ = torch.tensor([[0]], dtype=torch.float).to(params["device"])  # 负例的目标
    for epoch in range(params["train_scorer_epoch"]):
        print(f"正在训练第{epoch}个epoch:", end="", flush=True)
        train_loss = 0
        test_loss = 0
        for exp in train_exp_loader:  # 拿到经验，s是一整条轨迹
            # 每一个i是一个列表，里面是 [s1, s2, s3, s4]
            for idx in range(len(exp)-1):
                s = exp[idx]
                s_ = exp[idx+1]
                # 正例训练
                x = torch.cat([s, s_]).unsqueeze(0)  # 正例输入
                pred_y = model.forward(x)
                optimizer.zero_grad()
                loss = loss_fn(pred_y, y)
                loss.backward()
                optimizer.step()
                train_loss += loss

                # 反例训练
                x_ = torch.cat([s_, s]).unsqueeze(0)  # 正例输入
                pred_y_ = model.forward(x_)
                optimizer.zero_grad()
                loss = loss_fn(pred_y_, y_)
                loss.backward()
                optimizer.step()
                train_loss += loss
        # 每训练一个epoch，就测试
        with torch.no_grad():
            for exp in test_exp_loader:  # 拿到经验，s是一整条轨迹
                # 每一个i是一个列表，里面是 [s1, s2, s3, s4]
                for idx in range(len(exp)-1):
                    s = exp[idx]
                    s_ = exp[idx+1]
                    # 正例训练
                    x = torch.cat([s, s_]).unsqueeze(0)  # 正例输入
                    pred_y = model.forward(x)
                    loss = loss_fn(pred_y, y)
                    test_loss += loss

                    # 反例训练
                    x_ = torch.cat([s_, s]).unsqueeze(0)  # 正例输入
                    pred_y_ = model.forward(x_)
                    loss = loss_fn(pred_y_, y_)
                    test_loss += loss
        print(f"Train Loss: {train_loss:.4f}, eval loss: {test_loss:.4f}")
    torch.save(model.state_dict(),
               params["SCORER_PATH"])

if __name__ == "__main__":
    set_seed(params["seed"])
    main()
