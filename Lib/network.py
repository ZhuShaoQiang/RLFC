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
    

class FeedForward(BaseNetwork):
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
        self.network = nn.Sequential()

        # 从输入层到除了输出层之外的网络构造
        last_dim = self.input_dim
        for idx, h in enumerate(hidden):
            self.network.add_module(
                f"input_ff_{idx}", nn.Linear(last_dim, h)
            )
            self.network.add_module(
                f"activation_{idx}", activation
            )
            last_dim = h

        # 添加输出层
        self.network.add_module(
            "output", nn.Linear(last_dim, self.output_dim)
        )
        if not last_activation is None:
            self.network.add_module(
                "activation_last", last_activation,
            )
        
        ### 不删除的话，他会认为这个也是网络
        del self.activation

    def forward(self, x):
        """
        deterministic: 是否确定性产生动作，False是有随机
        """
        return self.network(x)
    
class CNN(BaseNetwork):
    """
    普通的卷积神经网络
    """
    hidden_dict = {
        "conv2d": nn.Conv2d,
        "maxpool": nn.MaxPool2d,
        "flatten": nn.Flatten,
    }
    def __init__(self, input_dim, activation: nn.Module=nn.LeakyReLU(),
                 hidden: list=[]) -> None:
        """
        input_dim: FF网络的输入维度
        output_dim: FF网络的输出维度
        
        hidden: 有多少隐藏层，每个隐藏层多少个神经元
        hidden中的参数
        str: 复杂网络

        last_activation: 输出层之后的激活函数，为None则为不要
        """
        super().__init__(input_dim=input_dim, output_dim=None, activation=activation)

        ### 构建网络
        self.network = nn.Sequential()
        print(f"拿到的：{hidden}")

        _idx_continue = len(hidden)-1
        for idx, val in enumerate(hidden):
            # 如果这里不是一个list，那么说明这里有问题
            # assert isinstance(val, list), f"hidden中存在{val}为不识别内容，请检查"

            # 构造卷积网络
            if isinstance(val, int):
                # 有共用的特征层
                _idx_continue = idx
                print(f"_idx_continue: {_idx_continue}")
                break
            elif val[0] == "activation":  # 如果是激活函数
                self.network.add_module(
                    f"activation_{idx}", activation
                )
            else:
                self.network.add_module(
                    val[0]+f"_{idx}", self.hidden_dict[val[0]](**val[1])
                )  # 添加这个模块

        # 得到目前为止，下面网络的输入形状
        self.latent_shape = self._get_conv_out_shape(input_dim)  # 得到计算的中间层的大小
        print(f"left: {hidden[_idx_continue:]}")
        print(f"hidden_len - 1: {len(hidden)-1}")
        last_dim = self.latent_shape[1]
        if _idx_continue < len(hidden):
            print(f"还有内容没有做")
            for idx, val in enumerate(hidden[_idx_continue:]):
                print(f"idx: {idx}, val: {val}")
                self.network.add_module(
                    f"input_share_ff_{idx}", nn.Linear(last_dim, val)
                )
                self.network.add_module(
                    f"activation_{idx}", activation
                )
                last_dim = val
        
        self.latent_shape = [1, last_dim]

        ### 删除一个变量
        del self.activation
        
    def _get_conv_out_shape(self, input_dim):
        """
        得到conv的输出形状，为dense层做准备
        """
        if len(self.network) <= 0:  # 说明没有卷积层
            return [1, self.input_dim]  # 返回输入层的形状
        # 这里说明network已经放入了东西，需要计算这个encoder网络的输出形状
        _zeros = torch.zeros([1,]+input_dim)
        return self.network(_zeros).shape
    
    def forward(self, x):
        """
        deterministic: 是否确定性产生动作，False是有随机
        """
        return self.network(x)


def _build_nn_from_JSON(
        in_shape, out_shape, 
        hidden: list, activation=nn.LeakyReLU,
        last_activation=None):
    """
    从一个参数设置里创建一个网络
    out_shape为None时，不构造输出层，说明这个部分是一个模块
    """
    ### 固有内容区
    str2net = {
        "conv2d": nn.Conv2d,
        "linear": nn.Linear,
        "flatten": nn.Flatten,
        "activation": activation,
    }
    def get_net_out_shape(net: nn.Module, in_shape):
        """
        得到一个网络的输入，in_shape是网络的输入
        """
        if isinstance(in_shape, int):
            in_shape = (in_shape)
        zeros = torch.zeros((1,)+in_shape)
        return net(zeros).shape
    ### 构造网络区
    accepted_in_shape_type = [tuple, int]
    assert type(in_shape) in accepted_in_shape_type, f"in_shape参数只接受tuple, int，你传来的是：{type(in_shape)}"
    cur_idx = -1
    layers = []
    for cur_idx, cur_hidden in enumerate(hidden):
        # 处理hidden
        if isinstance(cur_hidden, list):
            # 如果是个子列表, 是一个网络一个参数
            layer_name, layer_params = cur_hidden
            layers.append(
                str2net[layer_name](**layer_params)
            )
        elif isinstance(cur_hidden, int):
            # 如果是个int，说明
            # 1. 网络前面卷积层已经构造完成
            # 2. 网络只有linear层，要么是第一次构造，要么是前面liear层也构造的差不多了
            if cur_idx == 0:  # 这是第一层的linear
                assert isinstance(in_shape, int), f"线性层的in_shape必须是int，接收到了：{in_shape}, {type(in_shape)}"
                _linear_in = in_shape
                _linear_out = cur_hidden
            else: # 说明这不是第一次的linear了
                _tmp_net = nn.Sequential(*layers)  # 构造一个tmp网络
                # 这个shape是网络的输出shape
                # 目前位置，要么前面是conv结束了，那么肯定有flatten层
                # 要么前面全部是dense层，所以这个时候，shape必定是[B, xxx]
                # 所以只需要取最后一维就行了
                _linear_in = get_net_out_shape(_tmp_net, in_shape)[-1]
                _linear_out = cur_hidden
            layers.append(
                str2net["linear"](_linear_in, _linear_out)  # 构造linear层
            )
        else:  # 不知道什么类型
            raise f"请按照说明文档进行网络构造，不支持自定i的类型"
    
    ### 构造输出层
    if out_shape != None:
        _tmp_net = nn.Sequential(*layers)  # 构造一个tmp网络
        # 这个shape是网络的输出shape
        # 目前位置，要么前面是conv结束了，那么肯定有flatten层
        # 要么前面全部是dense层，所以这个时候，shape必定是[B, xxx]
        # 所以只需要取最后一维就行了
        _linear_in = get_net_out_shape(_tmp_net, in_shape)[-1]
        _linear_out = out_shape
        layers.append(
            str2net["linear"](_linear_in, _linear_out)  # 构造linear层
        )
    
    if last_activation != None:
        # 如果最后输出不是None的话，添加自定义最后一层的激活函数
        layers.append(
            last_activation()
        )

    return nn.Sequential(*layers)
    