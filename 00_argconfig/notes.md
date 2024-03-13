![](D:\College\projects\Build A Deep Learning Project Step by Step\00_argconfig\.assets\图片1.png)

# Unit Test

编写单元测试，有助于加深对项目各个模块的理解，减少代码出错的可能性。

下文是一个简单的单元测试参考框架：

```py
import unittest

class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # init data members
    
    def test1(self):
        # some test code
    
    def test2(self):
        # some test code
        
if __name__ == '__main__':
    unittest.main()
```



# Arguments

`argparse`库用于编写命令行接口，下面是使用该库的一个简单的代码示例

```py
# main.py
import argparse

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


if __name__ == '__main__':
    args, config = parse_option()
```



# YACS

YACS是一个轻量化的config库，config文件通常包含训练过程和模型本身的超参配置。配置并记录这些config有助于提供可重复的实验记录和结果。

基本的范式为`代码 + 某次实验的config (+ 其他可能的依赖项) = 可复现的实验结果`

> config能够记录实验对超参数的配置集，因此在修改代码时，不应该修改前面实验使用过的超参数分支，而是增加新的选择支。这样才能够保证代码的可回溯性。 

config的保存格式为YAML。



## 使用方法

首先需要创建config文件，惯用的命名为`config.py`。所有参数都会从该文件引用值，因此该文件的结构应当足够清晰且提供合适的默认值。

config文件的核心类为`CfgNode`，可以将其理解为一个config树中的节点。代码示例：

```py
# config.py
from yacs.config import CfgNode as CN # CfgNode是一个类似字典的容器，可以理解为一个config树中的一个节点，该容器可以用类似属性调用(即.xx)的方法来访问容器中的key

_C = CN()

# -----------------------------
# SYSTEM settings
# -----------------------------
_C.SYSTEM = CN() # 创建新的节点，通常一个节点对应argparser中的一个argument
_C.SYSTEM.NUM_GPUS = 8 # 节点下的值
_C.SYSTEM.NUM_WORKERS = 4

# -------------------------------
# TRAIN settings
# -------------------------------
_C.TRAIN = CN() # 创建新的节点
_C.TRAIN.BASE_LR = 0.1
_C.TRAIN.SCALES = (2, 4, 8, 16)

def get_config(args):
    '''get yacs CfgNode object with default values'''
    config = _C.clone() # 先拷贝所有的默认参数
    # 如果需要赋值，需要在cmd调用中传入对应的参数，例如
    # $python main.py --batch-size 64
    # config.defrost() # 解冻config
    # if args.batch_size:
    #	config.DATA.BATCH_SIZE = args.batch_size
    # config.freeze()
    
# 在main中调用
import argparse
parser = argparse.ArgumentParser()
args, unparsed = parser.parse_known_args()
config = get_config(args) # 这个config作为后续方法的传参即可
```



## 编写实验配置

在编写好配置文件后，对应于每次实验，编写YAML文件，例如

```py
# experiment.yaml
SYSTEM:
    NUM_GPUS: 2
TRAIN:
    SCALES: (1, 2)
```

使用时，从对应的yaml中匹配参数

```py
# config.py
def get_config(args):
	... # some parse options
    config = get_config_default()
    config.defrost()
    config.merge_from_file(args.config)
    config.freeze()  
```



## 练习

编写main.py和config文件，通过以下测试：

- argument测试：默认参数和传入参数，尝试通过传入参数定向到test.yaml
- config默认值和yaml配置值

```py
# test.py
from main import parse_option
import unittest
from train.logger import create_logger

class ArgConfigTests(unittest.TestCase):
    def test_args(self):
        logger.info(args)
        # check args
        self.assertTrue(args.config == './config/yaml/test.yaml')
        self.assertTrue(args.log == './log')
        
    def test_configs(self):
        logger.info(config)
        # check configs
        self.assertEqual(config.SYSTEM.LOG, './log')
        self.assertFalse(config.SYSTEM.CHECKPOINT == './checkpoint')
        self.assertEqual(config.SYSTEM.CHECKPOINT, './log')


if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger(args.log, name='argconfig_test.log')
    unittest.main() # run all tests
```

