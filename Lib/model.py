# -*- coding: utf-8 -*-

"""
这个文件存放各个模型的文件
如DQN网络
"""
from torch import nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    这个库内的模型的基本模型结构在这里体现
    """
    def __init__(self, state_dim, action_dim) -> None:
        """
        state_dim: 状态的形状
        action_dim: 动作的形状
        """
        super(BaseModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def forward(self, x, deterministic:bool = False):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

    @abstractmethod
    def evaluate_actions(self, x, action):
        """
        x: 输入的状态
        action: 旧的动作，这个函数计算之前的动作在当前的模型中，当前的观察下，的动作log_prob值，和当前的分布的entropy

        输出： value, action_log_prob, entropy
        """
        raise f"在 Base Model 中未实现这个方法 evaluate_actions，请在继承类中重写这个方法"
    
    @abstractmethod
    def get_state_value_only(self, x):
        """
        只得到评论家的价值

        输出：value
        """
        raise f"在 Base Model 中未实现这个方法 get_state_value_only，请在继承类中重写这个方法"
