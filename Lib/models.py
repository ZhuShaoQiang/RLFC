# -*- coding: utf-8 -*-

"""
这个文件存放各个模型的文件
如DQN网络
"""
import random

import torch
from torch import nn

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
    def _build_NN(self):
        """
        这是正交化网络参数的方法
        """
        raise f"在 Base Model 中未实现这个方法 _build_NN，在继承类中重写"
    
    @abstractmethod
    def forward(self, x, deterministic:bool = False):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"
    @abstractmethod
    def sample_action(self, dist):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

class BaseAC(BaseModel):
    """
    基础的演员评论家模型
    后面衍生的模型，全都是一个演员，一个评论家，演员可以产生动作,评论家需要给出价值
    """
    def __init__(self, input_dim, output_dim) -> None:
        """
        state_dim: 状态的形状
        action_dim: 动作的形状
        """
        super(BaseAC, self).__init__(input_dim, output_dim)
        self.actor = None
        self.critic = None

    def _build_NN(self):
        """
        AC的网络，应该有两个，一个actor，一个critic
        """
        assert self.actor != None, f"self.actor还未赋值网络"
        assert self.critic != None, f"self.critic还未赋值网络"
        initialize(self.actor)
        initialize(self.critic)
    
    @abstractmethod
    def forward(self, x):
        """
        x: 输入的状态

        输出：根据AC或者Q不同
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"
    @abstractmethod
    def sample_action(self, dist):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

class BaseQ(BaseModel):
    """
    基础的Q网络模型
    这里的模型，输出就是各个动作的价值
    """
    def __init__(self, input_dim, output_dim) -> None:
        """
        state_dim: 状态的形状
        action_dim: 动作的形状
        """
        super(BaseQ, self).__init__(input_dim, output_dim)
        self.action_space=output_dim
        self.nn = None

    def _build_NN(self):
        """
        对所有的网络正交初始化
        """
        assert self.nn != None, f"还未给self.nn赋值"
        initialize(self.nn)
    
    @abstractmethod
    def forward(self, x):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"

    @abstractmethod
    def sample_action(self, dist):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        raise f"在 Base Model 中未实现这个方法 forward，请在继承类中重写这个方法"


class RLFCScorer(BaseModel):
    """
    训练打分器的模型
    这个模型并不是网络，所以不需要继承BaseAC或者BaseQ
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
    
    def sample_action(self, dist):
        pass
            
class DQN(BaseQ):
    """
    自己想法的DQN版本，低维输入，低维输出
    """
    def __init__(self, input_dim, output_dim, activation: nn.Module=nn.LeakyReLU(),
                 hidden: list=[64, 64], last_activation: nn.Module=None) -> None:
        """
        input_dim: FF网络的输入维度
        output_dim: FF网络的输出维度
        
        hidden: 有多少隐藏层，每个隐藏层多少个神经元
        input -> hidden[0] -> hidden[...] -> output

        last_activation: 输出层之后的激活函数，为None则为不要
        """
        super().__init__(input_dim, output_dim)

        ### 构建网络
        self.nn = FeedForward(
            input_dim=input_dim, output_dim=output_dim, activation=activation,
            hidden=hidden, last_activation=last_activation
        )
        # self.Q = nn.Sequential()

        self._build_NN()

    def forward(self, x):
        """
        x: 输入的状态

        输出：action, value, action_log_prob
        """
        action_values = self.nn(x)
        # action_values是动作的预估价值
        return action_values
    
    def sample_action(self, action_values, epsilon: float=0.05):
        """
        选择动作，用epsilon贪心
        """
        if random.random() < epsilon:
            action = random.randint(0, self.action_space-1)
            action = torch.tensor(action)
        else:
            action = torch.argmax(action_values)
        return action