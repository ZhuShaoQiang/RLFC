# -*- coding: utf-8 -*-

from abc import ABC

from torch.utils.tensorboard import SummaryWriter

class Logger(ABC):
    """
    一个记录的对象，使用tensorboard
    """
    def __init__(self, path) -> None:
        """
        path: 记录日志的路径
        comment: 后缀，比如记录学习率等细节
        """
        super(Logger, self).__init__()
        self.path = path
        self._writter = SummaryWriter(self.path)

    def record_num(self, key, value, step):
        """
        记录
        key: 键：唯一的
        value: 值
        step: 横坐标轴，必须得有，并且需要递增
        """
        self._writter.add_scalar(
            key, value, step
        )
    
    def record_nums(self, key, kw_value, step):
        """
        记录多个数据
        key: 键，唯一的
        kw_value: 字典形式的值，如：{"agent1": 1, "agent2":1.2, "agent3":0.8}
        step: 横坐标轴，必须有，并且需要递增
        """
        self._writter.add_scalars(
            key,
            kw_value,
            step
        )
    
    def close(self):
        """
        关闭 
        """
        self._writter.close()
    
