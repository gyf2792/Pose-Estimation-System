# pnp位姿解算
import cv2
import numpy as np
import math

class PoseSolver:
    def __init__(self, K, D, obj_pts):
        self.K, self.D, self.obj_pts = K, D if D is not None else np.zeros(4), obj_pts

    def solve(self, img_pts):
        ok, rvec, tvec = cv2.solvePnP(self.obj_pts, np.array(img_pts, dtype=np.float64), self.K, self.D, flags=cv2.SOLVEPNP_IPPE)
        return (rvec, tvec) if ok else None

    def get_euler_angles(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy < 1e-6: return np.degrees([math.atan2(-R[1,2], R[1,1]), math.atan2(-R[2,0], sy), 0.0])
        return np.degrees([math.atan2(R[2,1], R[2,2]), math.atan2(-R[2,0], sy), math.atan2(R[1,0], R[0,0])])


    def compute_reprojection_error(self, img_pts, rvec, tvec):
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.D)
        proj = proj.reshape(-1, 2)
        img = np.array(img_pts).reshape(-1, 2)
        return float(np.linalg.norm(proj - img, axis=1).mean())
    # ==============================================

    def draw_axis(self, img, rvec, tvec, length=250):
        pts, _ = cv2.projectPoints(np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]), rvec, tvec, self.K, self.D)
        o, px, py, pz = [tuple(p.ravel().astype(int)) for p in pts[:4]]
        cv2.line(img, o, px, (0,0,255), 3); cv2.line(img, o, py, (0,255,0), 3); cv2.line(img, o, pz, (255,0,0), 3)