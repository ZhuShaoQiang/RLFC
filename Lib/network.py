# -*- coding: utf-8 -*-

"""
各种网络的代码文件
"""
import torch
import random
from torch import nn
from torch.distributions import Categorical

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

class DqnPolicy(BaseNetwork):
    """
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
        super().__init__(input_dim, output_dim, activation)

        ### 构建网络
        self.Q = nn.Sequential()
        self.action_space=output_dim

        # 从输入层到除了输出层之外的网络构造
        last_dim = self.input_dim
        for idx, h in enumerate(hidden):
            self.Q.add_module(
                f"input_ff_{idx}", nn.Linear(last_dim, h)
            )
            self.Q.add_module(
                f"activation_{idx}", activation
            )
            last_dim = h

        # 添加输出层
        self.Q.add_module(
            "output", nn.Linear(last_dim, self.output_dim)
        )
        if not last_activation is None:
            self.Q.add_module(
                "activation_last", last_activation,
            )
        
        self._build_NN()
    
    def _build_NN(self):
        """
        对所有的网络正交初始化
        """
        initialize(self.Q)

    def forward(self, x):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：action, value, action_log_prob
        """
        action_values = self.Q(x)
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


class MlpPolicy(BaseNetwork):
    """
    前馈神经网络，是密集网络
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
        super().__init__(input_dim=input_dim, output_dim=output_dim, activation=activation)

        ### 构建网络
        self.actor = nn.Sequential()
        self.critic = nn.Sequential()

        # 从输入层到除了输出层之外的网络构造
        last_dim = self.input_dim
        for idx, h in enumerate(hidden):
            self.actor.add_module(
                f"input_ff_{idx}", nn.Linear(last_dim, h)
            )
            self.actor.add_module(
                f"activation_{idx}", activation
            )

            self.critic.add_module(
                f"input_ff_{idx}", nn.Linear(last_dim, h)
            )
            self.critic.add_module(
                f"activation_{idx}", activation
            )

            last_dim = h

        # 添加输出层
        self.actor.add_module(
            "output", nn.Linear(last_dim, self.output_dim)
        )
        self.critic.add_module(
            "output", nn.Linear(last_dim, 1)
        )
        if not last_activation is None:
            self.actor.add_module(
                "activation_last", last_activation,
            )
        
        self._build_NN()
    
    def _build_NN(self):
        """
        对所有的网络正交初始化
        """
        initialize(self.actor)
        initialize(self.critic)
    
    def forward(self, x, deterministic: bool=False):
        """
        deterministic: 是否确定性产生动作，False是有随机
        """
        action_mean = self.actor(x)  # 产生一个动作的概率分布
        value = self.critic(x)

        # 弄成概率分布
        dist = Categorical(action_mean)

        if deterministic:  # 采样动作
            action = dist.mode
        else:
            action = dist.sample()
        
        # 计算动作的log_prob
        _log_prob = dist.log_prob(action)

        # 求和
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.mean(dim=1)
        else:
            action_log_prob = _log_prob.mean()

        return action, value, action_log_prob
            
    def evaluate_actions(self, states, actions):
        action_mean = self.actor(states)  # 产生一个动作的概率分布
        value = self.critic(states)

        # 弄成概率分布
        dist = Categorical(action_mean)
        # 计算动作的log_prob
        _log_prob = dist.log_prob(actions)
        # 求和
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.mean(dim=1)
        else:
            action_log_prob = _log_prob.mean()
        
        _entropy = dist.entropy()
        if len(_entropy.shape) > 1:
            entropy = _entropy.mean(dim=1)
        else:
            entropy = _entropy.mean()

        return value, action_log_prob, entropy
    