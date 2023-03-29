import argparse
import os
import os.path as osp
import torch
import time
import copy
import pickle
from yolox.exp import get_exp
from loguru import logger
from extractor import *
from segment import *
from recognition import *

def main():
    output_dir = "./demo/output/Outputvideos/"
    os.makedirs(output_dir, exist_ok=True)
    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(vis_folder, timestamp)
    
    save_root = './demo/output/'
    # seg
    video1_path = "./demo/output/Inputvideos/demo1.mp4"
    video2_path = "./demo/output/Inputvideos/demo2.mp4"

    # track
    track_result1 = track(video1_path, video_save_folder)
    track_result2 = track(video2_path, video_save_folder)

    silhouette1 = seg(video1_path, track_result1, save_root+'/silhouette/')
    silhouette2 = seg(video2_path, track_result2, save_root+'/silhouette/')

    # silhouette1 = getsil(video1_path, save_root+'/silhouette/')
    # silhouette2 = getsil(video2_path, save_root+'/silhouette/')

    # extract
    probe_feat = extract_sil(silhouette1, save_root+'/Gaitembs/')
    gallery_feat = extract_sil(silhouette2, save_root+'/Gaitembs/')

    # recognise
    pgdict1 = recognise_feat(probe_feat, gallery_feat)
    pgdict2 = recognise_feat(gallery_feat, probe_feat)

    # write the result back to the video
    writeresult(pgdict1, video1_path, video_save_folder)
    writeresult(pgdict2, video2_path, video_save_folder)


if __name__ == "__main__":
    main()
