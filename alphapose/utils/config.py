import yaml #yaml:一种标记语言，YAML 的语法和其他高级语言类似，并且可以简单表达清单、散列表，标量等数据形态。它使用空白符号缩进和大量依赖外观的特色，特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲（例如：许多电子邮件标题格式和YAML非常接近）
from easydict import EasyDict as edict #像访问属性一样访问dict里的变量


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))#yaml.load 读取yaml配置文件通过open方式读取文件数据，再通过load函数将数据转化为列表或字典
        return config #return yaml文件
