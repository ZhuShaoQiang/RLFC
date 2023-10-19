# -*- coding: utf-8 -*-

"""
使用PPO(AC)带有打分器的CarRacing程序
这个只是训练打分器的代码
"""

import sys
import os
sys.path.append(os.getcwd())
import random

import torch
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.utils import clip_grad_norm_

from Lib.algorithms import VanillaPPO
from Lib.models import RLFCScorer
from Lib.envwrappers import ImgToTensor
from Lib.utils import set_seed, compare_networks

from config.config_carracing_ppo_ac import params

from exp import initialize_exp, CARRACING_EXP

def main():
    """
    """
    os.makedirs(params["SAVE_PATH"], exist_ok=True)
    os.makedirs(params["LOGS_PATH"], exist_ok=True)

    ### 拿到经验
    train_exp_loader, test_exp_loader = initialize_exp(
        CARRACING_EXP, params, to_size=(96, 96)
    )
    ### 反例设置为从一个图片库里随便读取一张即可
    random_exp_img_path = "./exp/carracing_exp/allexp"
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    random_exp_dataset = ImageFolder(
        random_exp_img_path, transform=transform
    )  # 加载数据集
    random_exp_len = len(random_exp_dataset)
    # 创建dataloader
    dataloader = DataLoader(
        dataset=random_exp_dataset,
        batch_size=1,
        shuffle=True,
    )  # 一个随机的数据集加载器

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
    summary(scorer, input_size=(4, 96, 96))

    ### 开始训练
    # 配置正例负例的目标
    y_1 = torch.tensor([[1]], dtype=torch.float).to(params["device"])  # 正例的目标
    y_0 = torch.tensor([[0]], dtype=torch.float).to(params["device"])  # 负例的目标，为0
    # 配置训练函数
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=scorer.parameters(), lr=0.0001)
    for i in range(params["train_scorer_epoch"]):
        print(f"第{i}/{params['train_scorer_epoch']}轮训练:")
        # 四个连续状态
        for exps in train_exp_loader:
            # 每一条经验里有很多内容
            s = []
            for exp in exps:
                train_loss = 0
                if len(s) < 4:
                    s.append(exp)
                    continue
                s = s[-4:]  # 四个图片

                ### 设计正例输入，并训练
                s_1 = torch.cat(s).unsqueeze(0).to(params["device"])
                # s中的tensor都是1, 96, 96
                # 这样之后变成了4,96,96
                y_pred = scorer.forward(s_1)
                optim.zero_grad()
                loss = loss_fn(y_pred, y_1)
                loss.backward()
                clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optim.step()
                train_loss += loss

                ### 设计负例输入，并训练
                # 随机取出一个所有图片库之一的图片，作为下一个非专家经验
                random_idx_neg = random.randint(0, random_exp_len-1)
                random_sample, _ = dataloader.dataset[random_idx_neg]
                s_0 = torch.cat(s[:3]+[random_sample]).unsqueeze(0).to(params["device"])
                y_pred = scorer.forward(s_0)
                optim.zero_grad()
                loss = loss_fn(y_pred, y_0)
                loss.backward()
                clip_grad_norm_(scorer.parameters(), max_norm=1.0)
                optim.step()
                train_loss += loss
                print(f"{i}/{params['train_scorer_epoch']}: single step train loss: {train_loss}")


if __name__ == "__main__":
    set_seed(params["seed"])
    main()
