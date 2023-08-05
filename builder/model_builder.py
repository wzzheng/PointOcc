from model import *
from mmengine import build_from_cfg
from mmdet3d.registry import MODELS

def build(model_config):
    model = build_from_cfg(model_config, MODELS)
    model.init_weights()
    return model
