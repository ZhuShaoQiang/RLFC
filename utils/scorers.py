# -*- coding: utf-8 -*-

"""
打分器的文件
"""
import random

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.init import orthogonal_

from abc import ABC, abstractmethod

def initialize(net: nn.Module):
    """
    对网络进行正交初始化
    输入的参数需要是一个网络，如nn.Sequential
    """
    for p in net.parameters():
        if len(p.data.shape) >= 2:
            orthogonal_(p.data)


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
        if not isinstance(in_shape, tuple):
            in_shape = (in_shape,)
        zeros = torch.zeros((1,)+in_shape)
        return net(zeros).shape
    ### 构造网络区
    # accepted_in_shape_type = [tuple, int]
    # assert type(in_shape) in accepted_in_shape_type, f"in_shape参数只接受tuple, int，你传来的是：{type(in_shape)}"
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


class BaseModel(nn.Module, ABC):
    """
    这个库内的模型的基本模型结构在这里体现
    """
    def __init__(self, input_dim, output_dim, activation: nn.Module) -> None:
        """
        state_dim: 状态的形状
        action_dim: 动作的形状
        """
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

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
        super().__init__(input_dim, output_dim, activation=activation)
        self.__hidden = hidden
        self.__activation = activation
        self.__last_activation = last_activation

        self._build_NN(hidden)
    
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
    

class FeedForward(BaseModel):
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
                "activation_last", last_activation(),
            )
        
        ### 不删除的话，他会认为这个也是网络
        del self.activation
        self._build_NN()
    
    def _build_NN(self):
        initialize(self.network)

    def forward(self, x):
        """
        deterministic: 是否确定性产生动作，False是有随机
        """
        return self.network(x)