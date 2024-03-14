# main.py
import argparse
from config.config import get_config
from data.build import build_loader
from train.logger import create_logger

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def main():
    train_loader, test_loader, mixup_fn = build_loader(config)
    logger.info('finished data loading')

if __name__ == '__main__':
    args, config = parse_option()
    logger = create_logger('log', name='testlog.log')
    main()
    
    