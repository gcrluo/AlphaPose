import inspect #inspect模块是针对模块，类，方法，功能等对象提供些有用的方法。


class Registry(object):

    def __init__(self, name): #初始化
        self._name = name
        self._module_dict = dict() #定义的属性，是一个字典。用属性_module_dict 来保存config配置文件中的对应的字典数据所对应的class类

    def __repr__(self): #显示属性，自我描述。返回一个可以用来表示对象的可打印字符串，可以理解为java中的toString()。
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(#
            self._name, list(self._module_dict.keys()))#self.__class__.__name__，首先用self.__class__将实例变量指向类，然后再去调用__name__类属性
        return format_str

    @property  #把方法变成属性，通过self.name 就能获得name的值。
    def name(self):
        return self._name

    @property #同上，通过self.module_dict可以获取属性_module_dict，只读
    def module_dict(self):
        return self._module_dict

    def get(self, key): #普通方法，获取字典中指定key的value，_module_dict是一个字典，然后就可以通过self.get(key),获取value值
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        # 关键的一个方法，作用就是Register a module.
         # 将类送入了方法register_module()中执行。
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class): #判断是否为类，是的话为TRUE
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__ #获取类名
        if module_name in self._module_dict: #看改类是否已经登记在属性_module_dict中
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class#在module中dict新增key和value，key为类名，value为类对象

    def register_module(self, cls):#对上面的方法，修改了名字，添加了返回值，即返回类本身
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None): #这个cfg就是py配置文件中的字典
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    assert isinstance(default_args, dict) or default_args is None
    # 两个是断言，相当于判断，否的话抛出异常。
    args = cfg.copy() #两个是断言，相当于判断，否的话抛出异常。
    obj_type = args.pop('TYPE') #字典的pop作用：移除序列中key为‘type’的元素，并且返回该元素的值

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)  #获取obj_type的value。
        # 如果obj_type已经注册到注册表registry中，即在属性_module_dict中，则obj_type 不为None
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items(): #items()返回字典的键值对用于遍历
            args.setdefault(name, value)  #将default_args的键值对加入到args中，将模型和训练配置进行整合，然后送入类中返回
    return obj_cls(**args) #* *args是将字典unpack得到各个元素，分别与形参匹配送入函数中


def retrieve_from_cfg(cfg, registry):
    """Retrieve a module class from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        class: The class.
    """
    assert isinstance(cfg, dict) and 'TYPE' in cfg
    args = cfg.copy()
    obj_type = args.pop('TYPE')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    return obj_cls
