# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import argparse
import logging
import os
from types import MethodType

import torch

from .utils.config import update_config

parser = argparse.ArgumentParser(description='AlphaPose Training') #创建解析器
#argparse，命令行分析库，命令行选项、参数和子命令解析器

"----------------------------- Experiment options -----------------------------"
parser.add_argument('--cfg',
                    help='experiment configure file name', #实验设置文件
                    required=True,
                    type=str)   #add_argument() 方法，该方法用于指定程序能够接受哪些命令行选项。

parser.add_argument('--exp-id', default='default', type=str, #实验编号
                    help='Experiment ID')

"----------------------------- General options -----------------------------"
parser.add_argument('--nThreads', default=60, type=int, #数据安置方法
                    help='Number of data loading threads')
parser.add_argument('--snapshot', default=2, type=int, #快照，或许模型保存
                    help='How often to take a snapshot of the model (0 = never)')

parser.add_argument('--rank', default=-1, type=int, #分布式训练的节点秩
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://192.168.1.214:23345', type=str, #用于设置分布式培训的url
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, #分布式后端
                    help='distributed backend') #启动程序
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                    help='job launcher')

"----------------------------- Training options -----------------------------"
parser.add_argument('--sync', default=False, dest='sync', #同步
                    help='Use Sync Batchnorm', action='store_true')#Batchnorm是深度网络中经常用到的加速神经网络训练，加速收敛速度及稳定性的算法。
parser.add_argument('--detector', dest='detector', #目标探测
                    help='detector name', default="yolo")

"----------------------------- Log options -----------------------------"
parser.add_argument('--board', default=True, dest='board',
                    help='Logging with tensorboard', action='store_true')
parser.add_argument('--debug', default=False, dest='debug',
                    help='Visualization debug', action='store_true')
parser.add_argument('--map', default=True, dest='map',
                    help='Evaluate mAP per epoch', action='store_true')


opt = parser.parse_args()
cfg_file_name = os.path.basename(opt.cfg) #返回路径 path 的基本名称。
cfg = update_config(opt.cfg) #读取opt.cfg文件，读取为字典或列表

opt.world_size = cfg.TRAIN.WORLD_SIZE
cfg['FILE_NAME'] = cfg_file_name #opt.cfg路径的基本名称
cfg.TRAIN.DPG_STEP = [i - cfg.TRAIN.DPG_MILESTONE for i in cfg.TRAIN.DPG_STEP] #
opt.work_dir = './exp/{}-{}/'.format(opt.exp_id, cfg_file_name) #
opt.gpus = [i for i in range(torch.cuda.device_count())]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu") #GPU或者CPU


if not os.path.exists("./exp/{}-{}".format(opt.exp_id, cfg_file_name)):
    os.makedirs("./exp/{}-{}".format(opt.exp_id, cfg_file_name)) #没有此目录则创建目录

filehandler = logging.FileHandler(
    './exp/{}-{}/training.log'.format(opt.exp_id, cfg_file_name)) #训练日志
streamhandler = logging.StreamHandler()

logger = logging.getLogger('') #logger：日志对象，logging模块中最基础的对象，用logging.getLogger(name)方法进行初始化
logger.setLevel(logging.INFO) #设置日志等级
logger.addHandler(filehandler)
logger.addHandler(streamhandler)


def epochInfo(self, set, idx, loss, acc): #训练信息
    self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
        set=set, #
        idx=idx, #训练编号
        loss=loss, #损失
        acc=acc #准确度
    ))


logger.epochInfo = MethodType(epochInfo, logger) #MethodType可以把外部函数(方法)绑定到类或类的实例中
