#视频与参数IO管理
#内外参加载和极线校正
import cv2
import numpy as np
import os

def load_stereo_params(path):
    if not os.path.exists(path):
        return (None,) * 6
    data = np.load(path)
    return data['K_left'], data['dist_left'], data['K_right'], data['dist_right'], data['R'], data['T']

def init_rectify_maps(mtxL, distL, mtxR, distR, R_ext, T_ext, img_size):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, img_size, R_ext, T_ext.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_size, cv2.CV_16SC2)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_size, cv2.CV_16SC2)
    return mapL1, mapL2, mapR1, mapR2, P1, P2