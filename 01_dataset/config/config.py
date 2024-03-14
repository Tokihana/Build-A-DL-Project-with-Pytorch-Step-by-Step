# config.py
from yacs.config import CfgNode as CN

# root node
_C = CN()

## ----------------------------------------------
# SYSTEM settings
## ----------------------------------------------
_C.SYSTEM = CN()
# log path
_C.SYSTEM.LOG = './log'
# checkpoint path
_C.SYSTEM.CHECKPOINT = './checkpoint'

## ----------------------------------------------
# DATA settings
## ----------------------------------------------
_C.DATA = CN()
# datasets configs
# name of dataset, RAF-DB for default, supported: RAF-DB, AffectNet_7, AffectNet_8
_C.DATA.DATASET = 'RAF-DB'
# path to dataset
_C.DATA.DATA_PATH = '../datasets/RAF-DB'
# loader configs
# image size
_C.DATA.IMG_SIZE = 224
# batch size
_C.DATA.BATCH_SIZE = 64
# num of workers
_C.DATA.NUM_WORKERS = 8
# use pin memory or not
_C.DATA.PIN_MEMORY = True

## ----------------------------------------------
# MODEL settings
## ----------------------------------------------
_C.MODEL = CN()
## ----------------------------------------------
# TRAINING settings
## ----------------------------------------------

## ----------------------------------------------
# TEST settings
## ----------------------------------------------

def get_config_default():
    config = _C.clone()
    return config

def get_config(args):
    config = get_config_default()
    config.defrost()
    config.merge_from_file(args.config)
    config.freeze()
    return config