

# main() structure

`mian()`函数通常包含以下部分：

- data loader
- optimizer, scheduler
- model
- criterion
- training process
- test process

代码框架：

```py
def main():
    # load datasets
    train_loader, test_loader, mixup_fn = build_loader(config)
    logger.info('finished data loading')
    # create model, move to cuda, log parameters, flops
    model = None
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = model.flops()
    # build optimier
    optimizer = None
    # check if EVAL MODE or some other running MODE you need, such as THROUGHPUT_MODE
    if config.EVAL_MODE:
        load_weights(model, config.MODEL.RESUME)
        vaildate()
        return
    # build scheduler
    lr_scheduler = build_scheduler()
    # init criterion
    if config.AUG.MIXUP:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # whether needs to resume model?
    if config.TRAIN.RESUME:
        resume model
    
    # start training
    max_acc = 0.0
    logger.info('Start training')
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch()
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint()
        if val_loader is not None:
            validate()
            if max_accuracy updated:
                save_checkpoint
```



# utils

utils通常是一组与应用、任务无关的工具类。包括一些常用的评估指标、meter、模型保存与加载方法

因为`timm`提供了非常丰富的utils实现，所以很多时候是直接调用`timm`的



## configs

```py
## ----------------------------------------------
# MODEL settings
## ----------------------------------------------
_C.MODEL = CN()
# model name
_C.MODEL.ARCH = 'ir50'
# default nb_classes
_C.MODEL.NUM_CLASSES = 7
# resume checkpoint
_C.MODEL.RESUME = ''
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

## ----------------------------------------------
# MODE settings
## ----------------------------------------------
_C.MODE = CN()
_C.MODE.EVAL = False
```



## accuracy, Avgmeter

直接调用timm

```py
from timm.utils import accuracy, AverageMeter
```

## model save & load

只保存模型本体的简单框架

```py
# save
states = {'state_dict': model.state_dict()}
save_path = config.SYSTEM.CHECKPOINT + '.pth'
torch.save(states, save_path)
logger.info(f'saved checkpoint to {save_path}')
# load
checkpoint = torch.load(config.MODEL.RESUME)
model.load_state_dict(checkpoint['state_dict'])
logger.info(model)
```

保存运行时信息，方便断点续训的版本

```py
def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    logger.info(checkpoint)
    max_acc = 0.0
    if not config.MODE.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f'Loaded checkpoint from {checkpoint_path} successfully')
        if 'max_acc' in checkpoint:
            max_acc = checkpoint['max_acc'] 
    del checkpoint
    torch.cuda.empty_cache()
    return max_acc
    
def save_checkpoint(config, model, epoch, max_acc, optimizer, lr_scheduler, logger, is_best=False):
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_acc': max_acc,
        'epoch': epoch,
        'config': config,
    }
    if is_best:
        best_path = os.path.join(config.SYSTEM.CHECKPOINT + '_best.pth')
        torch.save(states, best_path)
    save_path = os.path.join(config.SYSTEM.CHECKPOINT + f'_epoch_{epoch}.pth')
    torch.save(states, save_path)
    logger.info(f'Save checkpoint to {save_path}')
```

## num_params

```py
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
```



## flops

调用thop库

```py
from thop import profile
def compute_param_flop():
    model = pyramid_trans_expr2()
    img = torch.rand(size=(1,3,224,224))
    flops, params = profile(model, inputs=(img,))
    print(f'flops:{flops/1024**3}G,params:{params/1024**2}M')
```

