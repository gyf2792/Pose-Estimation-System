import cv2
import numpy as np
import time
import csv
from datetime import datetime
import torch

from config import *
from io_manager import load_stereo_params, init_rectify_maps
from algorithms.traditional_detector import run_traditional, calc_roi
from algorithms.pose_solver import PoseSolver
from algorithms.stereo_solver import check_stereo_valid, triangulate_depth
from core.async_engine import AIResultBuffer, AIInferenceThread, FPSCounter
from core.tracker import KalmanTracker
from core.hallucinator import RigidBodyHallucinator
from core.filters import clamp_angles
from visualizer import draw_mode_banner, draw_info_panel, draw_keypoints
from model import MobilePose


def apply_hallucination(pts, hallucinator):
    """辅助判定脑补函数"""
    if pts is None: return None, None, False
    valid_mask = ~np.isnan(pts[:, 0])
    valid_count = np.sum(valid_mask)
    hallu_mask = ~valid_mask
    if valid_count == 4:
        return pts, hallu_mask, False
    elif valid_count >= 2:
        filled_pts, success = hallucinator.hallucinate(pts, valid_mask)
        if success:
            return filled_pts, hallu_mask, True
    return None, None, False


def get_repro_details(solver, img_pts, rvec, tvec):
    """计算重投影误差的均值及 8 个坐标偏移分量 (dx, dy)"""
    if rvec is None or tvec is None or img_pts is None:
        return None, [""] * 8

    proj, _ = cv2.projectPoints(solver.obj_pts, rvec, tvec, solver.K, solver.D)
    proj = proj.reshape(-1, 2)
    img = np.array(img_pts).reshape(-1, 2)

    err_vectors = (img - proj).ravel()
    mean_err = float(np.linalg.norm(img - proj, axis=1).mean())

    components = [round(float(v), 3) for v in err_vectors]
    return mean_err, components


def main():
    print("系统初始化中...")

    # 加载参数
    mtxL, distL, mtxR, distR, R_ext, T_ext = load_stereo_params(STEREO_PARAMS_PATH)
    capL, capR = cv2.VideoCapture(PATH_L), cv2.VideoCapture(PATH_R)

    if not capL.isOpened() or not capR.isOpened():
        print(" 无法打开视频文件，请检查 config.py 中的路径")
        return

    # 预读取一帧以获取正确的图像尺寸
    ret, sample = capL.read()
    if not ret:
        print("无法读取视频流")
        return
    capL.set(cv2.CAP_PROP_POS_FRAMES, 0)

    img_h, img_w = sample.shape[:2]
    img_size = (img_w, img_h)

    # 初始化极线校正映射
    mapL1, mapL2, mapR1, mapR2, P1, P2 = init_rectify_maps(mtxL, distL, mtxR, distR, R_ext, T_ext, img_size)

    print(f"加载 AI 模型 ({DEVICE})...")
    model = MobilePose(num_keypoints=4, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 实例化算法模块
    solver_L = PoseSolver(P1[:3, :3], None, TARGET_3D)
    solver_R = PoseSolver(P2[:3, :3], None, TARGET_3D)
    hallu_L = RigidBodyHallucinator(P1[:3, :3], None, TARGET_3D)
    hallu_R = RigidBodyHallucinator(P2[:3, :3], None, TARGET_3D)

    ai_buf = AIResultBuffer()
    fps_ai, fps_main = FPSCounter(), FPSCounter()
    ai_thread = AIInferenceThread(model, ai_buf, fps_ai)
    ai_thread.start()

    kf_L, kf_R = KalmanTracker(), KalmanTracker()

    # 初始化 CSV 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = open(f'pose_log_{timestamp}.csv', 'w', newline='', encoding='utf-8')
    csv_w = csv.writer(csv_file)
    header = ['frame', 'mode', 'fps_main', 'fps_ai', 'ai_lag', 'depth(mm)', 'pitch', 'yaw', 'roll', 'reproj_err',
              'dx1', 'dy1', 'dx2', 'dy2', 'dx3', 'dy3', 'dx4', 'dy4', 'cam_alive']
    csv_w.writerow(header)

    frame_idx = 0
    roi_L, roi_R = None, None
    current_mode = MODE_NORMAL
    trad_both_lost, cam_L_lost, cam_R_lost = 0, 0, 0

    dynamic_depth_offset, offset_initialized, smoothed_final_depth = 0.0, False, 0.0
    last_valid_angles = None

    print("开始处理视频流... (按 Q 退出)")

    try:
        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            cam_L_alive, cam_R_alive = retL and frameL is not None, retR and frameR is not None

            if not cam_L_alive and not cam_R_alive:
                print("视频播放结束")
                break

            frame_idx += 1
            t_main = time.time()

            # 极线校正
            rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR) if cam_L_alive else None
            rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR) if cam_R_alive else None

            # 更新 AI 推理帧
            if cam_L_alive and cam_R_alive:
                ai_thread.update_frames(rectL, rectR, frame_idx)
            elif cam_L_alive:
                ai_thread.update_frames(rectL, rectL, frame_idx)
            elif cam_R_alive:
                ai_thread.update_frames(rectR, rectR, frame_idx)

            grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY) if cam_L_alive else None
            grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY) if cam_R_alive else None

            # 传统检测流程
            final_L, is_kf_L, roi_L = run_traditional(grayL, kf_L, roi_L) if cam_L_alive else (None, False, None)
            final_R, is_kf_R, roi_R = run_traditional(grayR, kf_R, roi_R) if cam_R_alive else (None, False, None)

            # 模式切换逻辑
            cam_L_lost = (cam_L_lost + 1) if not cam_L_alive else 0
            cam_R_lost = (cam_R_lost + 1) if not cam_R_alive else 0
            cam_L_dead, cam_R_dead = cam_L_lost >= EXTREME_TRIGGER_FRAMES, cam_R_lost >= EXTREME_TRIGGER_FRAMES

            trad_L_ok, trad_R_ok = final_L is not None and not is_kf_L, final_R is not None and not is_kf_R
            trad_both_lost = (trad_both_lost + 1) if not (trad_L_ok and trad_R_ok) else 0

            if cam_L_dead or cam_R_dead:
                current_mode = MODE_EXTREME
            elif trad_both_lost >= FALLBACK_TRIGGER_FRAMES:
                current_mode = MODE_FALLBACK
            else:
                current_mode = MODE_NORMAL

            # 读取 AI 推理结果
            ai_pts_L, ai_pts_R, ai_age = ai_buf.read()

            vis_L = rectL.copy() if cam_L_alive else np.zeros((H, W, 3), np.uint8)
            vis_R = rectR.copy() if cam_R_alive else np.zeros((H, W, 3), np.uint8)

            raw_depth, pnp_depth_L, pnp_depth_R = 0.0, 0.0, 0.0
            angles, reproj_err = None, None
            err_components = [""] * 8
            used_pts_L, used_pts_R = None, None
            is_ai_pts = False
            hallu_mask_L, hallu_mask_R = None, None

            if current_mode in (MODE_FALLBACK, MODE_EXTREME):
                ai_pts_L, hallu_mask_L, is_hallu_L = apply_hallucination(ai_pts_L, hallu_L)
                ai_pts_R, hallu_mask_R, is_hallu_R = apply_hallucination(ai_pts_R, hallu_R)

            # 解算核心块
            if current_mode == MODE_NORMAL:
                if final_L is not None:
                    used_pts_L = final_L
                    res = solver_L.solve(final_L)
                    if res:
                        rvec, tvec = res
                        reproj_err, err_components = get_repro_details(solver_L, final_L, rvec, tvec)
                        if reproj_err < 10.0:
                            pnp_depth_L = float(tvec[2][0])
                            angles = clamp_angles(solver_L.get_euler_angles(rvec), last_valid_angles)
                            last_valid_angles = angles
                            solver_L.draw_axis(vis_L, rvec, tvec)
                            hallu_L.update_pose(rvec, tvec)
                        if final_R is not None and check_stereo_valid(final_L, final_R):
                            raw_depth = triangulate_depth(P1[:3], P2[:3], final_L, final_R)

                if final_R is not None:
                    used_pts_R = final_R
                    resR = solver_R.solve(final_R)
                    if resR and solver_R.compute_reprojection_error(final_R, resR[0], resR[1]) < 10.0:
                        pnp_depth_R = float(resR[1][2][0])
                        hallu_R.update_pose(resR[0], resR[1])

            elif current_mode == MODE_FALLBACK:
                roi_L, roi_R = calc_roi(ai_pts_L), calc_roi(ai_pts_R)
                if ai_pts_L is not None:
                    used_pts_L, is_ai_pts = ai_pts_L, True
                    res = solver_L.solve(ai_pts_L)
                    if res:
                        rvec, tvec = res
                        reproj_err, err_components = get_repro_details(solver_L, ai_pts_L, rvec, tvec)
                        if reproj_err < 10.0:
                            pnp_depth_L = float(tvec[2][0])
                            angles = clamp_angles(solver_L.get_euler_angles(rvec), last_valid_angles)
                            last_valid_angles = angles
                            solver_L.draw_axis(vis_L, rvec, tvec)
                            if not is_hallu_L: hallu_L.update_pose(rvec, tvec)
                        if ai_pts_R is not None and check_stereo_valid(ai_pts_L, ai_pts_R):
                            raw_depth = triangulate_depth(P1[:3], P2[:3], ai_pts_L, ai_pts_R)

                if ai_pts_R is not None:
                    used_pts_R = ai_pts_R
                    resR = solver_R.solve(ai_pts_R)
                    if resR and solver_R.compute_reprojection_error(ai_pts_R, resR[0], resR[1]) < 10.0:
                        pnp_depth_R = float(resR[1][2][0])
                        if not is_hallu_R: hallu_R.update_pose(resR[0], resR[1])

            else:  # MODE_EXTREME
                target_solver = solver_L if cam_L_alive else solver_R
                target_pts = ai_pts_L if cam_L_alive else ai_pts_R
                target_vis = vis_L if cam_L_alive else vis_R
                target_hallu = hallu_L if cam_L_alive else hallu_R
                is_hallu = is_hallu_L if cam_L_alive else is_hallu_R

                if target_pts is not None:
                    is_ai_pts = True
                    if cam_L_alive:
                        used_pts_L = target_pts
                    else:
                        used_pts_R = target_pts
                    res = target_solver.solve(target_pts)
                    if res:
                        rvec, tvec = res
                        reproj_err, err_components = get_repro_details(target_solver, target_pts, rvec, tvec)
                        if reproj_err < 10.0:
                            pnp_depth_L = float(tvec[2][0])
                            angles = clamp_angles(target_solver.get_euler_angles(rvec), last_valid_angles)
                            last_valid_angles = angles
                            target_solver.draw_axis(target_vis, rvec, tvec)
                            if not is_hallu: target_hallu.update_pose(rvec, tvec)

            # 深度补偿与滤波
            final_depth = 0.0
            if raw_depth > 0 and pnp_depth_L > 0:
                cur_off = raw_depth - pnp_depth_L
                dynamic_depth_offset = cur_off if not offset_initialized else 0.95 * dynamic_depth_offset + 0.05 * cur_off
                offset_initialized = True
                final_depth = raw_depth
            elif pnp_depth_L > 0:
                final_depth = pnp_depth_L + (dynamic_depth_offset if offset_initialized else 0)
            elif pnp_depth_R > 0:
                final_depth = pnp_depth_R + (dynamic_depth_offset if offset_initialized else 0)

            if final_depth > 0:
                if smoothed_final_depth == 0.0:
                    smoothed_final_depth = final_depth
                else:
                    smoothed_final_depth = 0.8 * smoothed_final_depth + 0.2 * np.clip(final_depth,
                                                                                      smoothed_final_depth - 50,
                                                                                      smoothed_final_depth + 50)

            # 可视化绘制
            if used_pts_L is not None: draw_keypoints(vis_L, used_pts_L, (is_kf_L and not is_ai_pts), is_ai_pts,
                                                      hallu_mask_L if is_ai_pts else None)
            if used_pts_R is not None: draw_keypoints(vis_R, used_pts_R, (is_kf_R and not is_ai_pts), is_ai_pts,
                                                      hallu_mask_R if is_ai_pts else None)

            fps_main.update(time.time() - t_main)
            combined = np.hstack((vis_L, vis_R))
            draw_mode_banner(combined, current_mode, fps_main.get_fps(), fps_ai.get_fps(), ai_age)
            draw_info_panel(combined, "LEFT CAM", smoothed_final_depth, angles, reproj_err,
                            is_predicted=(is_kf_L and not is_ai_pts), is_ai=is_ai_pts)

            # 写入 CSV 记录
            cam_alive_str = ("L" if cam_L_alive else "") + ("R" if cam_R_alive else "")
            csv_row = [
                frame_idx, current_mode, round(fps_main.get_fps(), 1), round(fps_ai.get_fps(), 1), ai_age,
                round(smoothed_final_depth, 2),
                round(angles[0], 2) if angles is not None else "",
                round(angles[1], 2) if angles is not None else "",
                round(angles[2], 2) if angles is not None else "",
                round(reproj_err, 2) if reproj_err is not None else ""
            ]
            csv_row.extend(err_components)
            csv_row.append(cam_alive_str)
            csv_w.writerow(csv_row)

            # 缩放显示窗口
            h_c, w_c = combined.shape[:2]
            cv2.imshow("Stereo Three-Mode Pose System", cv2.resize(combined, (1600, int(h_c * 1600 / w_c))))

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        # 资源释放
        ai_thread.stop()
        capL.release()
        capR.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"数据已保存至 {csv_file.name}")


if __name__ == "__main__":
    main()