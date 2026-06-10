#双目立体匹配
import cv2
import numpy as np
from config import STEREO_EPIPOLAR_THR

def check_stereo_valid(ptsL, ptsR):
    if ptsL is None or ptsR is None or len(ptsL) != 4 or len(ptsR) != 4: return False
    if np.max(np.abs(ptsL[:, 1] - ptsR[:, 1])) > STEREO_EPIPOLAR_THR: return False
    return not np.any((ptsL[:, 0] - ptsR[:, 0]) < 0)

def triangulate_depth(projL, projR, ptsL, ptsR):
    pts4D = cv2.triangulatePoints(projL, projR, ptsL.T.astype(np.float64), ptsR.T.astype(np.float64))
    return float(np.mean((pts4D[:3] / pts4D[3])[2]))