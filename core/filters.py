#滤波与限幅逻辑
import numpy as np

def clamp_angles(new_angles, old_angles, limit=5.0):
    if old_angles is None: return new_angles
    return tuple(float(np.clip(n, o - limit, o + limit)) for n, o in zip(new_angles, old_angles))