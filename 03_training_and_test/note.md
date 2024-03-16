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
