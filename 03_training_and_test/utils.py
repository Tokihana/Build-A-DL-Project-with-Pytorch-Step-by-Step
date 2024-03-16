# utils.py
# inline dependencies
import os
# third-party dependencies
import torch

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
        
        
def top1_accuracy(output, targets):
    output = torch.max(output, dim=1).values
    correct = output.eq(targets.reshape(-1, 1))
    acc = correct.float().sum() * 100. / targets.size(0)
    return acc