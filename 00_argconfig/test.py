# test.py
from main import parse_option
import unittest
from train.logger import create_logger

class ArgConfigTests(unittest.TestCase):
    def test_args(self):
        logger.info(args)
        # check args
        self.assertTrue(args.config == './config/yaml/test.yaml')
        self.assertTrue(args.log == './log')
        
    def test_configs(self):
        logger.info(config)
        # check configs
        self.assertEqual(config.SYSTEM.LOG, './log')
        self.assertFalse(config.SYSTEM.CHECKPOINT == './checkpoint')
        self.assertEqual(config.SYSTEM.CHECKPOINT, './log')


if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger(args.log, name='argconfig_test.log')
    unittest.main() # run all tests