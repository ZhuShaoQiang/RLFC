# -*- coding: utf-8 -*-

from .exp import ExpLoader
from .cliffwalking_exp.exp import generate_exps as cw_exp_generator
from .carracing_exp.exp import generate_exps as cr_exp_generator

CLIFFWALKING_EXP = "cliffwalking_exp"
CARRACING_EXP = "carracing_exp"

__exps = {
    CLIFFWALKING_EXP: cw_exp_generator,
    CARRACING_EXP: cr_exp_generator,
}

def initialize_exp(exp_name, params, split_ratio:float=0.3, to_size=(96, 96)) -> ExpLoader:
    """
    初始化经验
    exp_name: 初始化哪个经验
    params: 使用的参数
    split_ratio: 对测试机、训练集分割的比例，当本身分割的都有train和test时，这个参数默认无效

    返回值：需要的经验列表加载器
    """
    train_exp, test_exp = __exps[exp_name](**params, split_ratio=split_ratio, to_size=to_size)
    train_exploader = ExpLoader(exps=train_exp, device=params["device"])
    test_exploader = ExpLoader(exps=test_exp, device=params["device"])

    return train_exploader, test_exploader
