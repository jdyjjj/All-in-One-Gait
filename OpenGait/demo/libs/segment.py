import os 
import os.path as osp
import sys
import cv2
from pathlib import Path
import shutil
import torch
import math
import numpy as np
from tqdm import tqdm

from tracking_utils.predictor import Predictor
from yolox.utils import fuse_model, get_model_info
from loguru import logger
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from tracking_utils.visualize import plot_tracking, plot_track
from pretreatment import pretreat, imgs2inputs
sys.path.append((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
from seg_demo import seg_image
from infer import Predictor_opengait
from yolox.exp import get_exp

seg_cfgs = {  
    "model":{
        "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
    },
    "gait":{
        "dataset": "GREW",
    }
}

def imageflow_demo(video_path, track_result, sil_save_path):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (dict): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        Path: The directory of silhouette
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_video_name = video_path.split("/")[-1]

    save_video_name = save_video_name.split(".")[0]
    results = []
    ids = list(track_result.keys())

    # this is a time consuming operation
    predictor = Predictor_opengait(seg_cfgs["model"]["seg_model"])

    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id in ids and frame_id % 4 == 0:
                for tidxywh in track_result[frame_id]:
                    tid = tidxywh[0]
                    tidstr = "{:03d}".format(tid)
                    savesil_path = osp.join(sil_save_path, save_video_name, tidstr, "undefined")

                    x = tidxywh[1]
                    y = tidxywh[2]
                    width = tidxywh[3]
                    height = tidxywh[4]

                    x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                    w, h = x2 - x1, y2 - y1
                    x1_new = max(0, int(x1 - 0.1 * w))
                    x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                    y1_new = max(0, int(y1 - 0.1 * h))
                    y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(y2 + 0.1 * h))
                    
                    new_w = x2_new - x1_new
                    new_h = y2_new - y1_new
                    tmp = frame[y1_new: y2_new, x1_new: x2_new, :]

                    save_name = "{:03d}-{:03d}.png".format(tid, frame_id)
                    side = max(new_w,new_h)
                    tmp_new = [[[255,255,255]] * side] * side
                    tmp_new = np.array(tmp_new)
                    width = math.floor((side-new_w) / 2)
                    height = math.floor((side-new_h) / 2)
                    tmp_new[int(height):int(height + new_h),int(width):int(width + new_w),:] = tmp
                    tmp_new = tmp_new.astype(np.uint8)
                    tmp = cv2.resize(tmp_new, (192, 192))
                    seg_image(tmp, predictor, save_name, savesil_path)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    return Path(sil_save_path, save_video_name)

def seg(video_path, track_result, sil_save_path):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (Path): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        inputs (list): List of Tuple (seqs, labs, typs, vies, seqL) 
    """
    sil_save_path = imageflow_demo(video_path, track_result, sil_save_path)
    inputs = imgs2inputs(Path(sil_save_path), 64, False, seg_cfgs["gait"]["dataset"])
    return inputs

def getsil(video_path, sil_save_path):
    sil_save_name = video_path.split("/")[-1]
    inputs = imgs2inputs(Path(sil_save_path, sil_save_name.split(".")[0]), 
                64, False, seg_cfgs["gait"]["dataset"])
    return inputs
