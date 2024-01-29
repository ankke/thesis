import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from data.dataset_real_eye_vessels import build_real_vessel_network_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
import math

def build_mixed_data(config, mode='train', split=0.95, use_grayscale=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    config.DATA.DATA_PATH = config.DATA.SOURCE_DATA_PATH
    source_train_data = build_road_network_data(config, mode, split, config.DATA.NUM_SOURCE_SAMPLES, use_grayscale=config.DATA.TARGET_DATA_PATH != "mixed_road_dataset", domain_classification=0)

    config.DATA.DATA_PATH = config.DATA.TARGET_DATA_PATH
    if config.DATA.DATASET == "mixed_road_dataset":
        target_train_data = build_road_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)
    elif config.DATA.DATASET == "mixed_synthetic_eye_vessel_dataset":
        target_train_data = build_synthetic_vessel_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)
    elif config.DATA.DATASET == "mixed_real_eye_vessel_dataset":
        target_train_data = build_real_vessel_network_data(config, mode, split, config.DATA.NUM_TARGET_SAMPLES, use_grayscale, domain_classification=1)

    train_ds = ConcatDataset([source_train_data, target_train_data])

    return train_ds

    
