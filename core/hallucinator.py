#脑补刚体器
import cv2
import numpy as np


class RigidBodyHallucinator:
    def __init__(self, K, D, obj_pts):
        self.K = K
        self.D = D if D is not None else np.zeros(4)
        self.obj_pts = obj_pts
        self.last_rvec = None
        self.last_tvec = None

    def update_pose(self, rvec, tvec):
        """更新历史正确位姿，作为后续遮挡脑补的刚体先验约束"""
        self.last_rvec = rvec.copy()
        self.last_tvec = tvec.copy()

    def hallucinate(self, pts_2d, valid_mask):
        """
        基于上一帧的物理姿态，结合当前可见的特征点，脑补被遮挡的点。
        pts_2d: (4, 2) 可能包含 NaN 的浮点坐标数组
        valid_mask: (4,) 布尔数组，True表示点可见
        """
        if self.last_rvec is None or self.last_tvec is None:
            return pts_2d, False

        # 1. 用已知相机内参和最后一次完美姿态，把 855x1079 的矩形投射到 2D 像素空间
        proj_pts, _ = cv2.projectPoints(self.obj_pts, self.last_rvec, self.last_tvec, self.K, self.D)
        proj_pts = proj_pts.reshape(-1, 2)

        # 2. 计算当前实际看到的点，与投影出来的理想点之间的 2D 屏幕平移偏差 (dx, dy)
        dx, dy = 0.0, 0.0
        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            return pts_2d, False

        for i in range(4):
            if valid_mask[i]:
                dx += (pts_2d[i][0] - proj_pts[i][0])
                dy += (pts_2d[i][1] - proj_pts[i][1])
        dx /= valid_count
        dy /= valid_count

        # 3. 把这个位移补偿应用给那些看不见的点，形成严丝合缝的脑补矩形
        filled_pts = pts_2d.copy()
        for i in range(4):
            if not valid_mask[i]:
                filled_pts[i][0] = proj_pts[i][0] + dx
                filled_pts[i][1] = proj_pts[i][1] + dy

        return filled_pts, True

