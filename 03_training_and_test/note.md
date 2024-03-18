# training & test

- model加载
- optimizer
- schedular
- criterion
- train_one_epoch
- validate



## config

```py
# TRAINING settings
## ----------------------------------------------
_C.TRAIN = CN()
# epochs
_C.TRAIN.EPOCHS = 200
# start epoch
_C.TRAIN.START_EPOCH = 0
# warmup epochs
_C.TRAIN.WARMUP_EPOCHS = 20
# weight decay
_C.TRAIN.WEIGHT_DECAY = 0.05
# base lr
_C.TRAIN.BASE_LR = 3.5e-5
# warmup lr
_C.TRAIN.WARMUP_LR = 0.0
# min lr
_C.TRAIN.MIN_LR = 0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'consine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adam'
# epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# betas
_C.TRAIN.OPTIMIZER.BETAs = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.90.999
```



## model

model使用生成函数即可，这里以RepVGG为例

```py
model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint)
```

加载参数

```py
checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
model.load_state_dict(checkpoint)
```



## optimizer

```py
from torch import optim as optim

def build_optimizer(config, model):
    '''
    build optimizer, set weight decay
    '''
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip = model.no_weight_decay_keywords()
        
    parameters = set_weight_decay(model, skip, skip_keywords)
    
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    return optimizer

def set_weight_decay(model, skip_list, skip_keywords):
    '''
    ensure no decay was applied on frozen and skip param 
    '''
    has_decay=[]
    no_decay=[]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_key_words_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params':has_decay},
            {'params':no_decay, 'weight_decay':0.}]

def check_key_words_in_name(name, keywords=()):
    isin=False
    for keyword in keywords:
        if keyword in name:
            isin=True
            
    return isin
```

## Scheduler

```py
def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler
```



## criterion

```py
def build_criterion(config):
    criterion = None
    if config.TRAIN.CRITERION.NAME == 'SoftTargetCE':
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.TRAIN.CRITERION.NAME == 'LabelSmoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=config.TRAIN.CRITERION.LABEL_SMOOTHING)
    elif config.TRAIN.CRITERION.NAME == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    return criterion
```



## validate

因为是复现，优先跑验证。

validate不需要更新参数，所以要加上`torch.no_grad()`。

代码流程：

1. 创建`AvgMeter`，用于求loss和其他metrics的均值
2. 迭代val_lodaer
   - 计算loss和metric
   - AvgMeter
   - log
3. 返回loss和metric的均值



## train_one_epoch

流程如下：

1. `optimizer.zero_grad()`清理累积梯度
2. 迭代train_loader
   - 计算loss，求avg
   - `optimizer.zero_grad()`
   - `loss.backward()`
   - `optimizer.step()` 更新梯度
   - `scheduler.step()`调整lr

## train

在每个epoch，调用一次`train_one_epoch`和`validate`，并保存checkpoint，包括：

1. 模型的`state_dict`
2. optimizer的`state_dict`
3. scheduler的`state_dict`
4. 当前的max_mertric
5. 当前的epoch
6. 模型的相关配置





## finetune

finetune与train最大的区别在于，需要在运行train block之前，先做以下处理：

1. 读取checkpoint，并将与`num_class`的权重pop掉
2. freeze与`num_class`无关的权重

首先检查模型文件，确定和输出相关的part，并在预训练的checkpoint中检查keys，通过keys的名称进行筛选，例如寻找名称含有`linear`的权重

```py
checkpoint = torch.load(PATH)
has_linear = [key for key in checkpoint.keys() if 'linear' in key]
```

这个方法不一定有效，因为这里的名字是模型类成员的命名，如果模型的命名比较混乱，通过名称检索很可能找不到想要的权重。

以RepVGGplus为例，这个模型的输出相对复杂，除了最后的fc外，还对三个stage分别做了增强输出，这三个增强输出的fc没有特别命名，所以难以通过名称进行检索。

可以结合Netron的可视化进行分析，首先知道这三个`stage_aux`的结构为

```py
class RepVGGplus(nn.Module):
    def __init__(self):
        super(RepVGGplus, self).__init__()
        ...
        self.stage1_aux = self._build_aux_for_stage(self.stage1)
        self.stage2_aux = self._build_aux_for_stage(self.stage2)
        self.stage3_first_aux = self._build_aux_for_stage(self.stage3_first)
        ...
def _build_aux_for_stage(self, stage):
    stage_out_channels = list(stage.blocks.children())[-1].rbr_dense.conv.out_channels
    downsample = conv_bn_relu(in_channels=stage_out_channels, out_channels=stage_out_channels, kernel_size=3, stride=2, padding=1)
    fc = nn.Linear(stage_out_channels, self.num_classes, bias=True)
    return nn.Sequential(downsample, nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)
def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU())
    return result
```

以stage1_aux为例，其可视化如下：

![image-20240318114528799](C:\Users\wangj\AppData\Roaming\Typora\typora-user-images\image-20240318114528799.png)

结合模型结构以及数据集输出（`num_class = 1000`），可推断`stage1_aux.3`就是要找的fc层。
