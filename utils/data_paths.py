from glob import glob 
import os.path as osp
PROJ_DIR=osp.dirname(osp.dirname(__file__))
img_datas = glob(osp.join(PROJ_DIR, "data", "brain_pre_sam", "*", "*"))
all_classes = [
]

all_datasets = [
]
