# -*- coding: utf-8 -*-

"""
这个是用于加载经验的函数
这里对外暴露的是一个类，这个类可以用于加载经验，至于经验什么形式，那么使用者应该不在意
"""

import os
import torch
from torch.utils import data

class ExpLoader(data.Dataset):
    """
    经验的提取器
    """
    def __init__(self, exps, device) -> None:
        """
        一个经验的加载器
        exps: 经验的列表或者生成器
        """
        super().__init__()
        self.exps = exps
        self.device = device

    def __len__(self):
        """
        返回数据量长度
        """
        return len(self.exps)
    
    def __getitem__(self, idx):
        """
        这个地方是取数据主要的地方，取数据的逻辑从这里开始
        这里每次返回一整条轨迹的数据
        """
        if isinstance(self.exps, list):
            return torch.tensor(self.exps[idx], dtype=torch.float).to(self.device)
        elif isinstance(self.exps, data.Dataset):  # 目前为止只有两种，dataset或者是list
            return self.exps[idx]
        else:
            raise f"未识别的类型{type(self.exps)}, 期望：list、data.Dataset"
