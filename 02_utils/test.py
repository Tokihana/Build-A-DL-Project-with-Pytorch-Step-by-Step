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
from train.logger import create_logger
from model.linear import MLP
from utils import save_checkpoint

class UtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = MLP(4, 10, 3) # in_c = 4, out_c = 3
    def test_acc_avgmeter(self):
        target = torch.tensor([0, 1, 2, 3, 4]) # 5 examples, 5 class
        out1 = torch.tensor([[1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1]])
        out2 = torch.zeros((5, 5)) 
        logger.info(accuracy(out1, target, topk=(1,)))
        logger.info(accuracy(out2, target, topk=(1,)))
        acc2 = accuracy(out2, target, topk=(1,))
        self.assertEqual(acc2, [torch.tensor(20.)])
        
        accs = AverageMeter()
        accs.update(acc2[0], target.size(0)) # .update(val, n)
        accs.update(acc2[0], target.size(0))
        self.assertEqual(accs.avg, torch.tensor(20.))
        
    def test_model_save(self):
        inputs = torch.ones((5, 4))
        logger.info(self.model(inputs).shape)
        
        states = {'state_dict': self.model.state_dict()}
        torch.save(states, config.SYSTEM.CHECKPOINT + '.pth')
        self.assertTrue(os.path.exists(config.SYSTEM.CHECKPOINT + '.pth'))
        
    def test_model_load(self):
        checkpoint = torch.load(config.MODEL.RESUME)
        self.model.load_state_dict(checkpoint['state_dict'])
        logger.info(self.model)
        
        
    # def test_flops(self):
        # not emergency for now
        
    
    def test_num_params(self):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
    
    # def test_throughput(self):
        
    
    
        
        
        

if __name__ == '__main__':
    # make logger
    logger = create_logger('log', name='testlog.log')
    # make config
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='config yaml path')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    unittest.main()