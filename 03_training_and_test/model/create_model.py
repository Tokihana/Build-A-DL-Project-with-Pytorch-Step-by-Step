import torch
from .repvggplus import create_RepVGGplus_by_name
from .iresnet import iresnet50

def create_model(args, config):
    model = None
    if 'RepVGG' in config.MODEL.ARCH:
        if config.MODE.FINETUNE:
            model = create_finetune_RepVGG(args, config)
        else:
            model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint, config=config)
    elif config.MODEL.ARCH == 'IResNet50':
        if config.MODE.FINETUNE:
            model = create_finetune_IR50(config)
            #model = iresnet50(num_features=config.MODEL.NUM_CLASS)
        else:
            model = iresnet50()
        
    return model


def create_finetune_IR50(config):
    model = iresnet50(num_features=config.MODEL.NUM_CLASS)
    # load parameters
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    # pop num_class related params
    pop_list=[]
    for key in checkpoint.keys():
        if 'fc' in key or 'features' in key:
            pop_list.append(key)
    for key in pop_list:
        checkpoint.pop(key)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(missing, unexpected)
    return model
        
    
def create_finetune_RepVGG(args, config):
    model = create_RepVGGplus_by_name(config.MODEL.ARCH, deploy=False, use_checkpoint=args.use_checkpoint, config=config)
    # load checkpoint
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    # pop 
    return model