# -*- coding: utf-8 -*-

"""
这个文件存放各个模型的文件
如DQN网络
"""
from torch import nn
from torch.distributions import Categorical

from abc import ABC, abstractmethod

from .network import FeedForward
from .utils import initialize

class BaseModel(nn.Module, ABC):
    """
    这个库内的模型的基本模型结构在这里体现
    """
    def __init__(self, input_dim, output_dim) -> None:
        """
        state_dim: 状态的形状
        action_dim: 动作的形状
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x, deterministic:bool = False):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

class RLFCScorer(BaseModel):
    """
    训练打分器的模型
    输入时两个状态，输出是一个分数，代表这个和专家模型的相符程度
    训练的时候，目前确定的方案是：
    假设专家的经验是s1 s2 s3
    那么我们可以认为，非专家的经验是：s3 s2 s1
    """
    def __init__(self, input_dim, output_dim, activation: nn.Module=nn.LeakyReLU(),
                hidden: list=[64, 64], last_activation: nn.Module=None) -> None:
        super().__init__(input_dim, output_dim)

        self.nn = FeedForward(
            input_dim=input_dim, output_dim=output_dim, activation=activation,
            hidden=hidden, last_activation=last_activation
        )
    
        self._build_NN()
    
    def _build_NN(self):
        """
        对所有的网络正交初始化
        """
        initialize(self.nn)
    
    def forward(self, x):
        """
        这个只是个打分器，可以看做是一个回归模型
        """
        score = self.nn.forward(x)
        return score
            