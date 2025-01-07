import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from mono.mono_head import InferDAM
from stereo.stereo_head import InferCREStereo, InferGMStereo
from common.utils import imread_cv2, imread_pil


def workflow(left_path, right_path, mono_model_path, cres_stereo_model_path, gms_stereo_model_path):
    mono_worker = InferDAM()
    stereo_worker_gms = InferGMStereo()
    stereo_worker_cres = InferCREStereo()

    mono_worker.initialize(model_path=mono_model_path, encoder="vits")
    stereo_worker_gms.initialize(model_path=gms_stereo_model_path)
    stereo_worker_cres.initialize(model_path=cres_stereo_model_path)

    left = imread_cv2(left_path)
    right = imread_cv2(right_path)

    mono_out_disp = mono_worker.predict(image=left)
    gms_stereo_out_disp = stereo_worker_gms.predict(left, right)
    cres_stereo_out_disp = stereo_worker_cres.predict(left, right)

    print(f"mono disp shape is: {mono_out_disp.shape}")
    print(f"gms stereo disp shape is: {gms_stereo_out_disp.shape}")
    print(f"cres stereo disp shape is: {cres_stereo_out_disp.shape}")
    print("done!")




if __name__ == "__main__":
    left_path = "/home/william/extdisk/data/realsense-D455_depth_image/sample/infra1/1732083530.504292.png"
    right_path = "/home/william/extdisk/data/realsense-D455_depth_image/sample/infra2/1732083530.504292.png"
    mono_model_path = "/home/william/extdisk/checkpoints/depth-anything/depth_anything_v2_vits.pth"
    cres_stereo_model_path = "/home/william/extdisk/checkpoints/CREStereo/crestereo_eth3d.pth"
    gms_stereo_model_path = "/home/william/extdisk/checkpoints/gmstereo/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth"
    workflow(left_path, right_path, mono_model_path, cres_stereo_model_path, gms_stereo_model_path)



