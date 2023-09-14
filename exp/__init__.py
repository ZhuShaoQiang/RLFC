# -*- coding: utf-8 -*-

from .exp import ExpLoader
from .cliffwalking_exp.exp import generate_exps as cw_exp_generator
from .adventure_exp.exp import generate_exps as adventure_exp_generator

CLIFFWALKING_EXP = "cliffwalking_exp"
ADVENTURE_EXP = "adventure_exp"

__exps = {
    CLIFFWALKING_EXP: cw_exp_generator,
    ADVENTURE_EXP: adventure_exp_generator,
}

def initialize_exp(exp_name, params, split_ratio:float=0.3) -> ExpLoader:
    """
    初始化经验
    exp_name: 初始化哪个经验
    params: 使用的参数
    split_ratio: 对测试机、训练集分割的比例

    返回值：需要的经验列表加载器
    """
    train_exp, test_exp = __exps[exp_name](**params, split_ratio=split_ratio)
    train_exploader = ExpLoader(exps=train_exp, device=params["device"])
    test_exploader = ExpLoader(exps=test_exp, device=params["device"])

    return train_exploader, test_exploader
