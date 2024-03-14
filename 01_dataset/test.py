import unittest
import numpy as np
from train.logger import create_logger
import torch
from torchvision.transforms import v2
import torchvision
from data.build import build_loader, build_dataset, _get_rafdb_transform
import argparse
from config.config import get_config

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_fig = torch.ones((3, 14, 14))            
    
    def test_transform(self):
        logger.info(['fig size before transform:', self.test_fig.shape])
        #mean=[0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        transform, _ = _get_rafdb_transform()
        logger.info(['after transform:', transform(self.test_fig).shape])
        self.assertEqual(transform(self.test_fig).shape, torch.Size([3, 224, 224]))

    def test_dataset(self):
        train_dataset, val_dataset, num_classes = build_dataset(config=config)
        if config.DATA.DATASET == 'RAF-DB':
            self.assertEqual(train_dataset.__len__(), 12271)
            self.assertEqual(val_dataset.__len__(), 3068)
        elif config.DATA.DATASET == 'AffectNet_7':
            self.assertEqual(train_dataset.__len__(), 283901)
            self.assertEqual(val_dataset.__len__(), 3500)
        elif config.DATA.DATASET == 'AffectNet_8':
            self.assertEqual(train_dataset.__len__(), 287651)
            self.assertEqual(val_dataset.__len__(), 4000)
        elif config.DATA.DATASET == 'FERPlus':
            self.assertEqual(train_dataset.__len__(), 28386)
            self.assertEqual(val_dataset.__len__(), 3553)
        else:
            logger.info(f'DATASET {config.DATA.DATASET} NOT SUPPORTED')

    def test_dataloader_and_mixup(self):
        train_loader, val_loader, mix_fn = build_loader(config)
        for samples, targets in train_loader:
            break
        self.assertEqual(samples.shape, torch.Size([config.DATA.BATCH_SIZE, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]))
        logger.info(f'before mixup: {[samples.shape, targets.shape]}')
        samples, targets = mix_fn(samples, targets)
        logger.info(f'after mixup: {[samples.shape, targets.shape]}')
        

if __name__ == '__main__':
    # make logger
    logger = create_logger('log', name='testlog.log')
    # make config
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='config yaml path')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    unittest.main()