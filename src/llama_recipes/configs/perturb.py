import json
from omegaconf import OmegaConf

def load_yaml_config(config_filepath):
    config = OmegaConf.load(config_filepath)
    return config

def load_perturb_config_fromdict(config_filepath):
    config = OmegaConf.load(config_filepath)
    perturb_config = OmegaConf.create(config.perturb_config)
    return perturb_config