# -*- coding: utf-8 -*-

"""
各种网络的代码文件
"""
import torch
from torch import nn

from .utils import initialize

from abc import ABC, abstractmethod
from typing import Any

class BaseNetwork(nn.Module, ABC):
    """
    基础网络
    """
    def __init__(self, input_dim, output_dim, activation: nn.Module) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

    @abstractmethod
    def _build_NN(self):
        """
        对网络中的参数正交初始化
        """
        raise f"在 Base Model 中未实现这个方法 _build_NN，请在继承类中重写这个方法"

    @abstractmethod
    def forward(self, x):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

class Feedforward(BaseNetwork):
    """
    前馈神经网络，是密集网络
    """
    def __init__(self, input_dim, output_dim, activation: nn.Module=nn.LeakyReLU,
                 hidden: list=[64, 64], last_activation: nn.Module=None) -> None:
        """
        input_dim: FF网络的输入维度
        output_dim: FF网络的输出维度
        
        hidden: 有多少隐藏层，每个隐藏层多少个神经元
        input -> hidden[0] -> hidden[...] -> output

        last_activation: 输出层之后的激活函数，为None则为不要
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim, activation=activation)

        ### 构建网络
        self.NN = nn.Sequential()
        # 从输入层到除了输出层之外的网络构造
        last_dim = self.input_dim
        for idx, h in enumerate(hidden):
            self.NN.add_module(
                f"input_ff_{idx}", nn.Linear(last_dim, h)
            )
            self.NN.add_module(
                f"activation_{idx}", activation()
            )
            last_dim = h

        # 添加输出层
        self.NN.add_module(
            "output", nn.Linear(last_dim, self.output_dim)
        )
        if not last_activation is None:
            self.NN.add_module(
                "activation_last", last_activation()
            )
        
        self._build_NN()
    
    def _build_NN(self):
        """
        对所有的网络正交初始化
        """
        initialize(self.NN)
    
    def forward(self, x):
        res = self.NN(x)
        return res
            


    