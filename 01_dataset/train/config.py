# config.py
from yacs.config import CfgNode as CN

# base node
_C = CN()

# ------------------------------------------
# DATA settings
# -------------------------------------------
_C.DATA = CN()
# dataset name
_C.DATA.DATASET = 'RAF-DB' # 'RAF-DB' for now
# path to dataset
_C.DATA.DATA_PATH = '../datasets/RAF-DB'
# batch size
_C.DATA.BATCH_SIZE = 64
# input image size
_C.DATA.IMG_SIZE = 224
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8


def get_config(args):
    config = _C.clone()
    #config.defrost()
    #config.merge_from_list(args.config)
    #config.freeze()
    return config

def get_default_config():
    config = _C.clone()
    return config