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
from yolox.exp import get_exp

seg_cfgs = {  
    "model":{
        "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
        "ckpt" :    "./demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar",# 1
        "exp_file": "./demo/checkpoints/bytetrack_model/yolox_x_mix_det.py", # 4
    },
    "gait":{
        "dataset": "GREW",
    },
    "device": "gpu",
    "save_result": "True",
}


def loadckpt(exp):
    device = torch.device("cuda" if seg_cfgs["device"] == "gpu" else "cpu")
    model = exp.get_model().to(device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    ckpt_file = seg_cfgs["model"]["ckpt"]
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    logger.info("\tFusing model...")
    model = fuse_model(model)
    model = model.half()
    return model

exp = get_exp(seg_cfgs["model"]["exp_file"], None)
model = loadckpt(exp)

def track(video_path, video_save_folder):
    """Tracks person in the input video

    Args:
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """
    trt_file = None
    decoder = None
    device = torch.device("cuda" if seg_cfgs["device"] == "gpu" else "cpu")
    predictor = Predictor(model, exp, trt_file, decoder, device, True)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = video_path.split("/")[-1]
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results = []
    track_results={}
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if mark:
                        mark = False
                        diff = tid - 1
                    tid = tid - diff
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if frame_id not in track_results:
                            track_results[frame_id] = []
                        track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if seg_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if seg_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
    return track_results

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
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_video_name = video_path.split("/")[-1]

    save_video_name = save_video_name.split(".")[0]
    results = []
    ids = list(track_result.keys())
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id in ids:
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
                    tmp_new = [[[255,255,255]]*side]*side
                    tmp_new = np.array(tmp_new)
                    width = math.floor((side-new_w)/2)
                    height = math.floor((side-new_h)/2)
                    tmp_new[int(height):int(height+new_h),int(width):int(width+new_w),:] = tmp
                    tmp_new = tmp_new.astype(np.uint8)
                    tmp = cv2.resize(tmp_new,(192,192))
                    seg_image(tmp, seg_cfgs["model"]["seg_model"], save_name, savesil_path)

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1
    return Path(sil_save_path, save_video_name)

def writeresult(pgdict, video_path, video_save_folder):
    """Writes the recognition result back into the video

    Args:
        pgdict (dict): The id of probe corresponds to the id of gallery
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    """
    device = torch.device("cuda" if seg_cfgs["device"] == "gpu" else "cpu")
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, device, True)
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = video_path.split("/")[-1]
    save_video_path = save_video_name.split(".")[0]+ "-After.mp4"
    save_video_path = osp.join(video_save_folder, save_video_path)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    save_video_name = save_video_name.split(".")[0]

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_colors = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    if mark:
                        mark = False
                        diff = t.track_id - 1
                    track_id = t.track_id - diff

                    pid = "{}-{:03d}".format(save_video_name, track_id)
                    tid = pgdict[pid]
                    colorid = track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_colors.append(colorid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_track(
                    img_info['raw_img'], online_tlwhs, online_ids, online_colors, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if seg_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if seg_cfgs["save_result"] == "True":
        txtfile = "{}-{}".format(save_video_name, "After.txt")
        res_file = osp.join(video_save_folder, txtfile)
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

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
