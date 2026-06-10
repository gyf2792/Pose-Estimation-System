#卡尔曼滤波器
import cv2
import numpy as np


class KalmanTracker:
    def __init__(self):
        self.kfs = []
        for _ in range(4):
            kf = cv2.KalmanFilter(4, 2)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                            [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            self.kfs.append(kf)
        self.is_initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 15

    def init(self, pts):
        for i in range(4):
            self.kfs[i].statePost = np.array(
                [[pts[i][0]], [pts[i][1]], [0], [0]], np.float32)
        self.is_initialized = True
        self.lost_frames = 0

    def predict(self):
        preds = []
        for kf in self.kfs:
            p = kf.predict()
            preds.append([p[0, 0], p[1, 0]])
        return np.array(preds, dtype=np.float32)

    def correct(self, pts):
        for i in range(4):
            self.kfs[i].correct(np.array([[pts[i][0]], [pts[i][1]]], np.float32))
        self.lost_frames = 0

    def reset(self):
        self.is_initialized = False
        self.lost_frames = 0
