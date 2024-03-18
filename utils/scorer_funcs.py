# -*- coding: utf-8 -*-

"""
一些scorer的计算函数
"""

from typing import Any


def make_linear_scorer_func(coef):
    """
    线性计算函数
    """
    def linear_scorer_func(score) -> float:
        return coef * score
    return linear_scorer_func

def make_linear_decayable_scorer_func(coef, decay_step=500_000):
    """
    可衰减的打分器系数函数
    """
    class linear_decayable_scorer_func:
        def __init__(self) -> None:
            self.now_step = decay_step
            self.decay_step = decay_step
            self.coef = coef
        def __call__(self, score) -> float:
            """
            写一个可调用的方法，让这个对象作为一个func使用
            """
            ### 线计算衰减
            self.coef = (self.now_step / self.decay_step) * self.coef
            self.now_step -= 1
            if self.now_step <= 0:
                self.now_step = 0
            return self.coef * score
    return linear_decayable_scorer_func()
