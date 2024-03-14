# Blocks of loading data

首先需要确定，所在领域内常用的数据集，可以通过查阅领域内文献的datasets部分，并结合源码来了解。

一般来说，同个领域内，数据集的加载方式大差不差，只有一些细节、例如sampler、需要进行调整。

以表情识别为例，常用的数据集有：

- CK+
- AffectNet
- RAF-DB
- FERPlus
- CARE-S

加载数据集的模块可分为三个block

- dataset

  视觉分类任务用torchvision的比较多，有些数据集，例如ImageNet，已经内置在`torchvision.datasets`中了，可参考https://pytorch.org/vision/main/datasets.html

  对此之外的数据，可以使用`torchvision.datasets.DatasetFolder`或者其派生类`torchvision.datasets.ImageFolder`加载。

  默认的数据集组织结构如下，可通过重载`find_classes()`方法更改数据组织结构

  ```
  directory/
  ├── class_x
  │   ├── xxx.ext
  │   ├── xxy.ext
  │   └── ...
  │       └── xxz.ext
  └── class_y
      ├── 123.ext
      ├── nsdf3.ext
      └── ...
      └── asd932_.ext
  ```

- transform

  transform是一系列的图像处理方法，封装在`torchvision.transforms`里面，用于处理和增强图像。

  可以尝试使用`transforms.v2`，有MixUp这类增强方法的官方实现，一些常用transform的实现方法也更快。

  可参考[Getting started with transforms v2](https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py)。

- data_loader

  `torch.utils.data.DataLoader`将`dataset`和`sampler`结合，返回对应数据集的小批量迭代器。

  通常来讲需要关注的参数包括：`sampler, batch_size, shuffle, num_workers, pin_memory, drop_last`




## main utils

```py
# main.py
from data.build import build_loader	

def parse_option():
    # parse args and config

def main(config):
    train_loader, test_loader = build_loader(config)
    
if __name__ == '__main__':
    args, config = parse_option()
    main(config)
```



## configs

数据集加载常用的config如下，对应每个模块需要的参数：

```py
# config.py
## ----------------------------------------------
# DATA settings
## ----------------------------------------------
_C.DATA = CN()
# datasets configs
# name of dataset, RAF-DB for default
_C.DATA.DATASET = 'RAF-DB'
# path to dataset
_C.DATA.DATA_PATH = './datasets/RAF-DB'
# loader configs
# image size
_C.DATA.IMAGE_SIZE = 224
# batch size
_C.DATA.BATCH_SIZE = 64
# num of workers
_C.DATA.NUM_WORKERS = 8
# use pin memory or not
_C.DATA.PIN_MEMORY = True
```



## build_dataset

```py
def build_dataset(is_train, config)
```



## build_loader

```py
def build_loader(config)
```

