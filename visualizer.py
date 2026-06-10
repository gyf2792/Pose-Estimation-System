#可视化渲染模块
import cv2
import numpy as np
from config import *

def draw_keypoints(img, pts, is_predicted=False, is_ai=False, hallu_mask=None):
    if is_predicted:
        colors = [(255, 200, 0)] * 4
    elif is_ai:
        colors = [(255, 100, 255)] * 4
    else:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    for i, pt in enumerate(pts):
        if np.isnan(pt[0]): continue  # 跳过无效坐标
        x, y = int(pt[0]), int(pt[1])

        # 【修改点】如果该点是脑补出来的，用高亮的橙色标记
        if hallu_mask is not None and hallu_mask[i]:
            cv2.circle(img, (x, y), 8, (0, 165, 255), -1)
            cv2.circle(img, (x, y), 12, (0, 0, 255), 2)  # 外圈红色警告
            cv2.putText(img, str(i) + "(B)", (x + 15, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.circle(img, (x, y), 8, colors[i], -1)
            cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(img, str(i), (x + 12, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)


def draw_mode_banner(img, mode: int, fps_main: float, fps_ai: float, ai_age: int):
    h, w = img.shape[:2]
    bg = img.copy()
    cv2.rectangle(bg, (0, 0), (w, 62), (30, 30, 30), -1)
    cv2.addWeighted(bg, 0.75, img, 0.25, 0, img)

    name = MODE_NAMES[mode]
    color = MODE_COLORS[mode]

    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    tx = (w - tw) // 2
    cv2.putText(img, name, (tx, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    fps_c = (0, 255, 0) if fps_main >= 25 else (0, 200, 255)
    cv2.putText(img, f"Main FPS: {fps_main:.1f}", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_c, 2)

    ai_label = f"AI FPS: {fps_ai:.1f}  lag:{ai_age}f"
    ai_c = (0, 255, 100) if ai_age <= 3 else (0, 165, 255) if ai_age <= 8 else (0, 0, 255)
    (tw2, _), _ = cv2.getTextSize(ai_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, ai_label, (w - tw2 - 12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ai_c, 2)


def draw_info_panel(img, side_label: str, depth: float, angles,
                    reproj_err, pts=None, is_predicted=False, is_ai=False,
                    top_offset=70):
    """通用信息面板（统一展示一个 Depth 和位姿角，不再区分 PnP 或 Stereo）"""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (10, top_offset), (500, top_offset + 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    title_c = (255, 100, 255) if is_ai else (0, 255, 255)
    tag = " [AI]" if is_ai else " [Trad]"
    if is_predicted:
        tag += " (KF)"
    cv2.rectangle(img, (10, top_offset), (500, top_offset + 38), (50, 50, 50), -1)
    cv2.putText(img, side_label + tag, (18, top_offset + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, title_c, 2)

    y = top_offset + 58
    if depth > 0:
        cv2.putText(img, f"Depth: {depth / 1000:.3f} m",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 255), 2);
        y += 30

    if angles is not None:
        cv2.putText(img, f"Pitch: {angles[0]:.1f} deg",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2);
        y += 28
        cv2.putText(img, f"Yaw:   {angles[1]:.1f} deg",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2);
        y += 28
        cv2.putText(img, f"Roll:  {angles[2]:.1f} deg",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2);
        y += 28

    if reproj_err is not None:
        cv2.putText(img, f"Reproj Err: {reproj_err:.2f} px",
                    (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2);
        y += 28

    if depth == 0:
        cv2.putText(img, "Target Lost", (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if pts is not None:
        draw_keypoints(img, pts, is_predicted=is_predicted, is_ai=is_ai)

