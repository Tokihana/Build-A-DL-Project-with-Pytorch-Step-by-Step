import unittest
import numpy as np
from logger import create_logger
import torch
from torchvision.transforms import v2
import torchvision
from data.build import build_dataset
import argparse
from train.config import get_config

class ConfigTests(unittest.TestCase):
    def test_args(self):
        logger.info('Parsed args:')
        logger.info(args)
        return

    def test_config_getting(self):
        logger.info(config)
        
    def test_default(self):
        assertEqual(self.config.DATASET, 'RAF-DB')
        logger.info('Config Default Value Passed')
        return

    def test_modified(self):
        #assertFalse(self.config.BATCH_SIZE == 64)
        #assertFalse(self.config.BATCH_SIZE == 16)
        return

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_fig = torch.ones((3, 14, 14))            
    
    def test_transform(self):
        logger.info(['fig size before transform:', self.test_fig.shape])
        #mean=[0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        transform = v2.Compose([
            v2.Resize((224, 224)),
            #v2.ToTensor(),
            #v2.Normalize(mean, std),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        logger.info(['after transform:', transform(self.test_fig).shape])
        self.assertEqual(transform(self.test_fig).shape, torch.Size([3, 224, 224]))

    def test_dataset(self):
        # dataset = build_dataset(is_train=True, config=self.config)
        return

    def test_dataloader(self):
        return

    def test_mixup(self):
        return
        

if __name__ == '__main__':
    # make logger
    logger = create_logger('log', name='testlog.log')
    # make config
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument('--config', default='./configs/test.yaml', type=str, help='config yaml path')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    unittest.main()