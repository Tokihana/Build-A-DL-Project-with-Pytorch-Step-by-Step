# main.py
# inline dependencies
import argparse
# third-party dependencies
from timm.utils import accuracy, AverageMeter
# local dependencies
from config.config import get_config
from data.build import build_loader
from train.logger import create_logger
from utils import save_checkpoint, load_checkpoint

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

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
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    # whether needs to resume model?
    max_acc = 0.0
    if config.TRAIN.RESUME:
        max_acc = load_checkpoint(config=config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
    
    # start training
    
    logger.info('Start training')
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch()
        if epoch % config.SYSTEM.SAVE_FREQ == 0:
            save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger)
        if val_loader is not None:
            validate()
            if max_accuracy updated:
                save_checkpoint(config=config, model=model, epoch=epoch, max_acc=max_acc, optimizer=optimizer, lr_scheduler=lr_scheduler, logger=logger, is_best=True)
    

if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger('log', name='testlog.log')
    main()
    
    