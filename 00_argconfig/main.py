# main.py
import argparse
from config.config import get_config

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./config/yaml/test.yaml', type=str, help='path to config yaml')
    parser.add_argument('--log', default='./log', type=str, help='path to log')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


if __name__ == '__main__':
    args, config = parse_option()