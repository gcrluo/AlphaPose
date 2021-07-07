from torch import nn

from alphapose.utils import Registry, build_from_cfg, retrieve_from_cfg #

#构建SPPE、LOSS、DATASET三个注册类实例
SPPE = Registry('sppe') #single-person pose estimator (SPPE)单人姿态估计
LOSS = Registry('loss') #注册了一个对象名为LOSS ，属性name为  loss 的对象
DATASET = Registry('dataset')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]   #build_from_cfg()返回值是一个带形参的类，返回时也就完成了实例化的过程。
        return nn.Sequential(*modules)  #所以modules就是一个class类的列表
    # nn.Sequential 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_sppe(cfg, preset_cfg, **kwargs): #建立sppe
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, SPPE, default_args=default_args)


def build_loss(cfg):建立损失
    return build(cfg, LOSS)


def build_dataset(cfg, preset_cfg, **kwargs): #建立数据
    exec(f'from ..datasets import {cfg.TYPE}')
    default_args = {
        'PRESET': preset_cfg,
    }
    for key, value in kwargs.items():
        default_args[key] = value
    return build(cfg, DATASET, default_args=default_args)


def retrieve_dataset(cfg): #数据检索
    exec(f'from ..datasets import {cfg.TYPE}')
    return retrieve_from_cfg(cfg, DATASET)
