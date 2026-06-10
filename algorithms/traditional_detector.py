import cv2
import numpy as np
from itertools import combinations
from config import *


def calc_roi(pts, margin=ROI_MARGIN):
    if pts is None or len(pts) == 0: return None
    valid_pts = [p for p in pts if not np.isnan(p[0])]
    if not valid_pts: return None
    pts_arr = np.array(valid_pts)
    min_x, min_y = np.min(pts_arr, axis=0)
    max_x, max_y = np.max(pts_arr, axis=0)
    return (max(0, int(min_x - margin)), max(0, int(min_y - margin)),
            int(max_x - min_x + 2 * margin), int(max_y - min_y + 2 * margin))


def verify_target_pattern_fast(img_gray, center, radius):
    try:
        cx, cy = int(center[0]), int(center[1])
        h, w = img_gray.shape
        margin = int(radius * 1.5)
        if cx - margin < 0 or cx + margin >= w or cy - margin < 0 or cy + margin >= h: return False
        if radius < RADIUS_MIN or radius > RADIUS_MAX: return False
        inner_r = int(radius * 0.3)
        if inner_r < 3: return False
        angs = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        sx, sy = (cx + inner_r * np.cos(angs)).astype(int), (cy + inner_r * np.sin(angs)).astype(int)
        samp = img_gray[sy, sx]
        lo, hi = float(samp.min()), float(samp.max())
        mid, contrast = (lo + hi) / 2, hi - lo
        if contrast < CONTRAST_THRESHOLD * (0.6 if mid < 70 else 1.0): return False
        sig = (samp > mid).astype(int)
        return int(np.sum(np.abs(np.diff(np.concatenate([sig, [sig[0]]]))))) == 4
    except:
        return False


def detect_targets_traditional(img_gray, roi=None):
    all_dets = []
    ox, oy = (roi[0], roi[1]) if roi else (0, 0)
    proc = img_gray[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] if roi else img_gray
    try:
        thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(proc, (3, 3), 0), 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 1. 寻找所有可能的检测点
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (AREA_MIN <= area <= AREA_MAX): continue
            (lx, ly), radius = cv2.minEnclosingCircle(cnt)
            if radius > 0 and (area / (np.pi * radius ** 2)) >= CIRCULARITY_MIN:
                pt = np.array([lx + ox, ly + oy], dtype=np.float32)
                if verify_target_pattern_fast(img_gray, pt, radius):
                    all_dets.append({'pt': pt, 'radius': radius})

        # 2. 去重逻辑 (注意这里的缩进，必须在 for 循环完全结束之后执行)
        if len(all_dets) > 1:
            filtered = []
            for det in all_dets:
                rep = -1
                for idx, ex in enumerate(filtered):
                    if np.linalg.norm(det['pt'] - ex['pt']) < 15:
                        if det['radius'] > ex['radius']:
                            rep = idx
                        break
                if rep >= 0:
                    filtered[rep] = det
                else:
                    filtered.append(det)
            return filtered

        return all_dets
    except Exception as e:
        return []


def sort_points_traditional(pts_list):
    if not pts_list or len(pts_list) < 4: return None
    pts_list = sorted(pts_list, key=lambda x: x['radius'], reverse=True)[:8]
    best_quad, best_score = None, -1
    for combo in combinations(range(len(pts_list)), 4):
        pts = np.array([pts_list[i]['pt'] for i in combo])
        center = np.mean(pts, axis=0)
        ps = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
        area = cv2.contourArea(pts)
        if area > best_score: best_score, best_quad = area, ps
    if best_quad is None: return None
    return np.roll(best_quad, -np.argmin(best_quad[:, 0] + best_quad[:, 1]), axis=0)


def run_traditional(gray, kf_tracker, current_roi):
    detections = detect_targets_traditional(gray, roi=current_roi)
    sorted_pts = sort_points_traditional(detections)

    if sorted_pts is None and current_roi is not None:
        current_roi = None
        detections = detect_targets_traditional(gray, roi=None)
        sorted_pts = sort_points_traditional(detections)

    is_predicted = False
    predicted = kf_tracker.predict() if kf_tracker.is_initialized else None

    if sorted_pts is not None:
        if not kf_tracker.is_initialized:
            kf_tracker.init(sorted_pts)
        else:
            kf_tracker.correct(sorted_pts)
        final_pts = sorted_pts
    else:
        if kf_tracker.is_initialized and kf_tracker.lost_frames < kf_tracker.max_lost_frames:
            final_pts = predicted
            kf_tracker.lost_frames += 1
            is_predicted = True
        else:
            final_pts = None
            kf_tracker.is_initialized = False

    new_roi = calc_roi(final_pts, ROI_MARGIN)

    return final_pts, is_predicted, new_roi