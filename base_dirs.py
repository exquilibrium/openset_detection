import os

BASE_DIR_FOLDER = os.path.dirname(os.path.realpath(__file__))
BASE_WEIGHTS_FOLDER = BASE_DIR_FOLDER+'/mmdetection/weights'
if BASE_DIR_FOLDER == '/home/chen/openset_detection':
    BASE_DATA_FOLDER = '/media/chen/76AECF8EAECF4579/data'
else: # '/home/chen_le/openset_detection' >>> DLR
    BASE_DATA_FOLDER = '/volume/hot_storage/slurm_data/chen_le/ARCHES'
BASE_RESULTS_FOLDER = BASE_DIR_FOLDER + '/results'
BASE_PRETRAINED_FOLDER = BASE_DIR_FOLDER + '/pretrained'
