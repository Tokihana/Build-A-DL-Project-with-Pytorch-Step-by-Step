import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torchsampler import ImbalancedDatasetSampler

def build_loader(config):
    config.defrost()
    train_dataset, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    val_dataset, _ = build_dataset(is_train=True, config=config)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler,
                                               shuffle=True,
                                               batch_size=config.DATA.BATCH_SIZE,
                                               num_workers=config.DATA.NUM_WORKERS,
                                               pin_memory=config.DATA.PIN_MEMORY,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=config.DATA.BATCH_SIZE,
                                             num_workers=config.DATA.NUM_WORKERS,
                                             pin_memory=config.DATA.PIN_MEMORY,
                                             drop_last=False)

    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    return train_loader, val_loader, cutmix_or_mixup

def build_dataset(is_train, config):
    # RAF-DB
    if config.DATA.DATASET == 'RAF-DB':
        mean=[0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if is_train:
            transform = v2.Compose([
                v2.Resize((224, 224)), # Resize() only accept PIL or Tensor type images
                v2.RandomHorizontalFlip(),
                v2.ToTensor(), # warned by Torch: ToTensor() will be removed in a future release, use [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] instead
                                # if the param scale is True, the range of the inputs will be normalized
                                # note that ToImage() only accept the inputs with legth 3
                v2.Normalize(mean, std),
                v2.RandomErasing(scale=(0.02, 0.25)),
            ])
            dataset = datasets.ImageFolder(config.DATA.DATA_PATH, transform)
        else: 
            transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToTensor(),
                v2.Normalize(mean, std),
            ])
            dataset = datasets.ImageFolder(config.DATA.DATA_PATH, transform)
        nb_classes = 7
    else:
        raise NotImplementError("DATASET NOT SUPPORTED")
    return dataset, nb_classes
