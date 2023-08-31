# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn.init import orthogonal_
import torch.cuda

def initialize(net: nn.Module):
    """
    对网络进行正交初始化
    输入的参数需要是一个网络，如nn.Sequential
    """
    for p in net.parameters():
        if len(p.data.shape) >= 2:
            orthogonal_(p.data)


def set_seed(seed):
    '''set seed for cuda 11.x'''

    # 基础设置，random库和numpy库的随机种子
    random.seed(seed)
    np.random.seed(seed)

    # 禁止hash随机化
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 消耗显存24MB，消除乱序multi-stream execution的影响
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # pytorch seed常规设置
    torch.manual_seed(seed) # CPU设置
    torch.cuda.manual_seed(seed) # GPU设置
    torch.cuda.manual_seed_all(seed)  # 多GPU训练

    # 消除卷积不确定性
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 强制pytorch使用确定性算法
    # 使用的算子没有确定的实现则会报错，如：torch.Tensor.index_add_()、gather、scatter等
    # 可以尝试升级pytorch版本，发现bug可以直接提PR给torch
    # https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    torch.use_deterministic_algorithms(True)

    torch.set_printoptions(threshold=5000, linewidth=120)
