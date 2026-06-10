import cv2
import numpy as np
import torch
from config import AI_CONF_THRESH


def preprocess_for_model(img, input_size):
    t = cv2.resize(img, input_size).astype(np.float32) / 255.0
    return ((t - 0.45) / 0.225).transpose(2, 0, 1)


def taylor_refine(heatmap, coord):
    h, w = heatmap.shape
    px, py = int(coord[0]), int(coord[1])
    if px < 2 or px >= w - 2 or py < 2 or py >= h - 2:
        return coord

    dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
    dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
    dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
    dyy = 0.25 * (heatmap[py + 2][px] - 2 * heatmap[py][px] + heatmap[py - 2][px])
    dxy = 0.25 * (heatmap[py + 1][px + 1] - heatmap[py + 1][px - 1]
                  - heatmap[py - 1][px + 1] + heatmap[py - 1][px - 1])

    deriv = np.array([[dx], [dy]])
    hessian = np.array([[dxx, dxy], [dxy, dyy]])

    if np.linalg.det(hessian) != 0:
        off = (-np.linalg.inv(hessian) @ deriv).ravel()
        if abs(off[0]) < 1.5 and abs(off[1]) < 1.5:
            return [coord[0] + off[0], coord[1] + off[1]]

    return coord


def refine_with_opencv(img, rough_coord, window_size=30):
    x, y = int(rough_coord[0]), int(rough_coord[1])
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x - window_size), max(0, y - window_size)
    x2, y2 = min(w_img, x + window_size), min(h_img, y + window_size)
    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        return rough_coord

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    guess = np.array([[[x - x1, y - y1]]], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 15, 0.005)

    try:
        ref = cv2.cornerSubPix(gray, guess, (5, 5), (1, 1), crit)
        lx, ly = ref[0][0]
        fx, fy = x1 + lx, y1 + ly
        if abs(fx - x) > window_size or abs(fy - y) > window_size:
            return rough_coord
        return (fx, fy)
    except Exception:
        return rough_coord


def decode_heatmap_to_pts(heatmaps_np, h_orig, w_orig, frame):
    bs, nj, h, w = heatmaps_np.shape
    flat = heatmaps_np.reshape(bs, nj, -1)
    idx, maxvals = np.argmax(flat, 2), np.amax(flat, 2)
    final = []

    for i in range(4):
        if maxvals[0][i] < AI_CONF_THRESH:
            final.append([np.nan, np.nan])
            continue

        px, py = idx[0][i] % w, np.floor(idx[0][i] / w)
        rx, ry = taylor_refine(heatmaps_np[0][i], [px, py])
        final.append(refine_with_opencv(frame, (rx * (w_orig / w), ry * (h_orig / h))))

    return np.array(final, dtype=np.float32)