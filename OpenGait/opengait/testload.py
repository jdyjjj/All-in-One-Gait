import argparse
import os
import os.path as osp
import pickle
import time
import cv2
import torch
from modeling import models
from main import initialization
from utils import config_loader

torch.distributed.init_process_group('nccl', init_method='env://')


def loadModel(model_type, cfg_path):
    Model = getattr(models, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    #model._load_ckpt(savepath)
    return model


cfgs = {  "gaitmodel":{
    "model_type": "Baseline",
    "cfg_path": "OpenGait/configs/baseline/baseline_OUMVLP.yaml",
},
}
print("========= Loading model..... ==========")
initialization(config_loader(cfgs["gaitmodel"]["cfg_path"]), False)
gaitmodel = loadModel(**cfgs["gaitmodel"])
gaitmodel.requires_grad_(False)
gaitmodel.eval()
print("========= Load Done.... ==========")