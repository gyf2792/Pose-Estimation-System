#全局配置中心
import numpy as np
import torch

# ==================== 路径配置 ====================
PATH_L = r'C:\Users\DELL\Desktop\Graduation Project\new_begining\inter_external_paras\video - 副本\cam0\阵面转动版.avi'
PATH_R = r'C:\Users\DELL\Desktop\Graduation Project\new_begining\inter_external_paras\video - 副本\cam1\阵面转动版.avi'
STEREO_PARAMS_PATH = r'C:\Users\DELL\Desktop\Graduation Project\new_begining\inter_external_paras\stereo_params_3.6_ratio_Global.npz'
MODEL_PATH = "weights03/best_model03.pth"

# ==================== 物理参数 ====================
W, H = 855, 1079
half_w, half_h = W / 2.0, H / 2.0
TARGET_3D = np.array([
    [-half_w, -half_h, 0], [half_w, -half_h, 0],
    [half_w, half_h, 0], [-half_w, half_h, 0]
], dtype=np.float32)

# ==================== 传统检测与 ROI ====================
AREA_MIN, AREA_MAX = 45, 4000
CIRCULARITY_MIN = 0.50
RADIUS_MIN, RADIUS_MAX = 5, 80
CONTRAST_THRESHOLD = 30
ROI_MARGIN = 80

# ==================== AI 模型参数 ====================
INPUT_SIZE = (512, 512)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AI_CONF_THRESH = 0.35

# ==================== 阈值与模式 ====================
STEREO_EPIPOLAR_THR = 5.0
FALLBACK_TRIGGER_FRAMES = 1
EXTREME_TRIGGER_FRAMES = 30

MODE_NORMAL, MODE_FALLBACK, MODE_EXTREME = 1, 2, 3
MODE_NAMES = {
    MODE_NORMAL: "MODE 1  NORMAL   [Trad+Trad Stereo]",
    MODE_FALLBACK: "MODE 2  FALLBACK [AI+AI   Stereo]",
    MODE_EXTREME: "MODE 3  EXTREME  [Mono AI PnP]",
}
MODE_COLORS = {MODE_NORMAL: (0, 255, 100), MODE_FALLBACK: (0, 200, 255), MODE_EXTREME: (0, 80, 255)}