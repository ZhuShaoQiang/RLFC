# -*- coding: utf-8 -*-

"""
这个文件存放各个模型的文件
如DQN网络
"""
import random

import torch
from torch import nn
from torch.distributions import Normal

from abc import ABC, abstractmethod

from .network import FeedForward, CNN, _build_nn_from_JSON
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
    def forward(self, x):
        """
        x: 输入的状态
        deterministic: 是否确定性的取值

        输出：根据网络的不同这里不同
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
        self.encoder = None
        self.actor = None
        self.critic = None
        self.sigma = nn.Parameter(
            torch.zeros(self.output_dim), requires_grad=True
        )

    def _build_NN(self):
        """
        AC的网络，应该有两个，一个actor，一个critic
        """
        assert self.actor != None, f"self.actor还未赋值网络"
        assert self.critic != None, f"self.critic还未赋值网络"
        if self.encoder != None:
            initialize(self.encoder)
        initialize(self.critic)
        initialize(self.actor)
    
    @abstractmethod
    def forward(self, x):
        """
        x: 输入的状态

        输出：根据continous和discrete不同而不同
        """
        raise f"在 BaseAC 中未实现这个方法 forward，请在继承类中重写这个方法"
    @abstractmethod
    def sample_action(self, action_mean):
        """
        action_mean: forward产生的分布

        输出：action, action_log_prob
        """
        raise f"在 BaseAC 中未实现这个方法 sample_action，请在继承类中重写这个方法"

    @abstractmethod
    def evaluate_action(self, x, action):
        """
        x: 状态
        action: 动作

        得到这个动作在当前的x中的action_log_prob，以及这个dist的entropy
        """
        raise f"在 BaseAC 中为实现这个方法 evaluate_action，请在继承类中重写这个方法"
    @abstractmethod
    def get_state_value_only(self, x):
        """
        x: 状态
        只计算状态对应的价值
        """
        raise f"在 BaseAC中未实现这个方法 get_state_value_only，请重写这个方法"

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
    def __init__(self, input_dim, output_dim, activation: nn.Module=nn.LeakyReLU,
                hidden: list=[64, 64], last_activation: nn.Module=None) -> None:
        super().__init__(input_dim, output_dim)
        self.__hidden = hidden
        self.__activation = activation
        self.__last_activation = last_activation

        self._build_NN(hidden)

        # self.nn = FeedForward(
        #     input_dim=input_dim, output_dim=output_dim, activation=activation,
        #     hidden=hidden, last_activation=last_activation
        # )
    
    def _build_NN(self, ortho: bool=True):
        """
        ortho: 是否对所有的网络正交初始化
        hidden_bias: hidden层向后偏移的数量
        给模型传一个hidden，然后这个函数根据这个hiddenbuild一个网络模型
        """
        # 找到第一个int类型
        self.nn = _build_nn_from_JSON(
            self.input_dim, self.output_dim,
            self.__hidden, self.__activation, self.__last_activation)

        if ortho:
            initialize(self.nn)
    
    def forward(self, x):
        """
        这个只是个打分器，可以看做是一个回归模型
        """
        score = self.nn.forward(x)
        return score
    
    def sample_action(self, dist):
        pass

class AC(BaseAC):
    """
    普通的PPO
    hidden里面不仅仅要使用int，要使用字符串做神经网络
    """
    def __init__(self, input_dim, action_dim, activation: nn.Module=nn.LeakyReLU(),
                 hidden: list=[64, 64], hidden_bias=0, last_activation: nn.Module=None) -> None:
        """
        last_activation只给actor网络，不需要给别人
        """
        super().__init__(input_dim, action_dim)

        # 找到第一个int类型
        idx_split = len(hidden) - 1  # 这样的话，如果没有int型的话，就默认全是cnn了
        for idx, val in enumerate(hidden):
            if val == "split":
                idx_split = idx
                break  # 找到分割地方
        
        idx_split += hidden_bias

        # encoder网络
        if idx_split != 0:
            self.encoder = CNN(
                input_dim=input_dim, activation=activation,
                hidden=hidden[:idx_split]
            )
            dense_input = self.encoder.latent_shape[1]
        else:
            self.encoder = None
            dense_input = input_dim

        # actor网络
        self.actor = FeedForward(
            input_dim=dense_input, output_dim=action_dim, activation=activation,
            hidden=hidden[idx_split:], last_activation=last_activation,
        )
        # critic网络，输出是一个价值，肯定是1
        self.critic = FeedForward(
            input_dim=dense_input, output_dim=1, activation=activation,
            hidden=hidden[idx_split:], last_activation=None,
        )

        self._build_NN()
    
    def forward(self, x):
        """
        x: 输入的状态

        输出：动作的分布或者价值，这个就是dist
        """
        # 经过encoder
        if self.encoder != None:
            latent = self.encoder.forward(x)
        else:
            latent = x
        
        # 计算结果
        action_mean = self.actor.forward(latent)
        value = self.critic.forward(latent)
        return action_mean, value

    def sample_action(self, action_mean):
        """
        action_mean就是actor网络产生的分布

        输出：action, action_log_prob
        """
        # 根据产生得动作的分布，得到一个分布
        dist = Normal(action_mean, torch.exp(self.sigma))
        # 采样得到动作
        action = dist.sample()

        # 得到动作的log_prob
        _log_prob = dist.log_prob(action)
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.sum(dim=1)
        else:
            action_log_prob = _log_prob.sum()
        
        return action, action_log_prob
    
    def evaluate_action(self, x, action):
        """
        x: 状态
        action: 动作

        得到这个动作在当前的x中的action_log_prob，以及这个dist的entropy
        """
        action_mean, value = self.forward(x)
        # 根据产生得动作的分布，得到一个分布
        dist = Normal(action_mean, torch.exp(self.sigma))
        # 采样得到动作
        action = dist.sample()

        # 得到动作的log_prob
        _log_prob = dist.log_prob(action)
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.sum(dim=1)
        else:
            action_log_prob = _log_prob.sum()
        
        _entropy = dist.entropy()
        if len(_entropy.shape) > 1:
            entropy = _entropy.sum(dim=1)
        else:
            entropy = _entropy.sum()

        return value, action_log_prob, entropy
    
    @torch.no_grad()
    def get_state_value_only(self, x):
        if self.encoder != None:
            latent = self.encoder.forward(x)
        else:
            latent = x
        
        value = self.critic.forward(latent)
        return value

            
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

"""
从ATLA里面直接拷贝过来的，看看效果
"""
class CnnAC(nn.Module, ABC):
    def __init__(self, state_dim, action_dim, feature_dim: int=512) -> None:
        """
        state_dim: 输入图像的形状，如[3, 84, 84]
        action_dim: 输出动作的个数，如：3

        feature_dim: 每个图像应该被编码为多大的特正，如512
        """
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        super(CnnAC, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )  # 这个是卷积网络，用来提取图像的特征的

        self.latent_shape = self._get_conv_output_shape(tuple(state_dim))
        del self.Encoder  # 删除这个，重新创建
        self.Encoder = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.latent_shape[1], feature_dim),
            nn.ReLU(),
        )

        self.sigma = nn.Parameter(
            torch.zeros(self.action_dim), requires_grad=True
        )
        self.Actor_net = nn.Sequential(
            nn.Linear(feature_dim, self.action_dim),
        )  # 连续空间中，这个就不必要了
        self.Value_net = nn.Sequential(
            nn.Linear(feature_dim, 1),
        )

        # 正交初始化网络
        initialize(self.Encoder)
        initialize(self.Value_net)
        initialize(self.Actor_net)

    def _get_conv_output_shape(self, input_shape):
        """
        计算卷积层的输出形状
        input_shape: 如[3, 84, 84]
        """
        zeros = torch.zeros((1,)+input_shape)
        return self.Encoder(zeros).shape
    
    def forward(self, x):
        """
        如果noise不是None，说明这个是加到latent上的noise
        """
        latent = self.Encoder(x)  # 得到特征

        action_mean = self.Actor_net(latent)
        value = self.Value_net(latent)

        return action_mean, value

    def sample_action(self, action_mean):
        """
        action_mean就是actor网络产生的分布

        输出：action, action_log_prob
        """
        # 根据产生得动作的分布，得到一个分布
        dist = Normal(action_mean, torch.exp(self.sigma))
        # 采样得到动作
        action = dist.sample()

        # 得到动作的log_prob
        _log_prob = dist.log_prob(action)
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.sum(dim=1)
        else:
            action_log_prob = _log_prob.sum()
        
        return action, action_log_prob
    
    def evaluate_action(self, x, action, noise=None):
        latent = self.Encoder(x)
        if not noise is None:
            latent = latent + noise

        action_mean = self.Actor_net(latent)
        value = self.Value_net(latent)

        dist = Normal(action_mean, torch.exp(self.sigma))
        _log_prob = dist.log_prob(action)  # 计算log_prob
        if len(_log_prob.shape) > 1:
            action_log_prob = _log_prob.sum(dim=1)
        else:
            action_log_prob = _log_prob.sum()
        
        _entropy = dist.entropy()
        if len(_entropy.shape) > 1:
            entropy = _entropy.sum(dim=1)
        else:
            entropy = _entropy.sum()

        return value, action_log_prob, entropy
    
    def get_state_value_only(self, x, noise=None):
        latent = self.Encoder(x)
        if not noise is None:
            latent = latent + noise
        value = self.Value_net(latent)  # 得到这个状态的价值
        return value
