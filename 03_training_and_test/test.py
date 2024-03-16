# test.py
# inline dependencies
import os
import unittest
import argparse
# third-party dependencies
import torch
import torch.nn as nn
from torchvision.transforms import v2
from timm.utils import accuracy, AverageMeter
# local dependencies
from data.build import build_loader, build_dataset, _get_rafdb_transform
from config.config import get_config
from train import create_logger, build_optimizer, build_scheduler, build_criterion
from model import create_model
from utils import save_checkpoint, load_checkpoint

from main import validate, train_one_epoch

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/repvgg.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

class RepVGG(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.train_loader, self.val_loader, self.mix_fn = build_loader(config)
        self.model = create_model(args, config)
        self.checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        # logger.info(checkpoint.keys())
        if not config.MODE.FINETUNE:
            self.model.load_state_dict(self.checkpoint)
        self.model.cuda()
        self.optimizer = build_optimizer(config, self.model)
        self.scheduler = build_scheduler(config, self.optimizer, len(self.train_loader))
        self.criterion = build_criterion(config)
        
    def test_model(self):
        logger.info('MODEL INFO:')
        logger.info(self.model)
        logger.info(self.model.parameters())
        
    def test_optimizer(self):
        logger.info('OPTIMIZER INFO:')
        logger.info(self.optimizer)
        
    def test_scheduler(self):
        logger.info('SCHEDULER INFO:')
        logger.info(self.scheduler)
    
    def test_criterion(self):
        logger.info('CRITERION INFO:')
        logger.info(self.criterion)
        
    def test_load_non_strict_checks(self):
        new_dict = self.checkpoint.copy()
        pop_list = []
        for key in new_dict.keys():
            if 'linear' in key:
                pop_list.append(key)
        logger.info(pop_list)
        for key in pop_list:
            new_dict.pop(key) # can not pop during dict iteration
            
    def test_train_one_epoch(self):
        train_one_epoch(config=config, model=self.model, data_loader=self.train_loader, epoch=0, mix_fn = self.mix_fn, criterion=self.criterion, optimizer=self.optimizer, lr_scheduler=self.scheduler, logger=logger)
        acc, loss = validate(config, self.model, self.val_loader, logger)
        
    def test_validate(self):
        self.model.cuda()
        #acc, loss = validate(config, self.model, self.val_loader, logger)
        

if __name__ == '__main__':
    # make logger
    logger = create_logger('log', name='testlog.log')
    # make config
    args, config = parse_option()
    
    unittest.main()