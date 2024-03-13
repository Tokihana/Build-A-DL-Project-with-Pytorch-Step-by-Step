# config.py
from yacs.config import CfgNode as CN

# root node
_C = CN()

## ----------------------------------------------
# SYSTEM settings
## ----------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.LOG = './log'
_C.SYSTEM.CHECKPOINT = './checkpoint'

## ----------------------------------------------
# DATASET settings
## ----------------------------------------------

## ----------------------------------------------
# MODEL settings
## ----------------------------------------------

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