#异步推理引擎与共享缓存
import threading
import time
import torch
import numpy as np
from collections import deque
from algorithms.ai_detector import preprocess_for_model, decode_heatmap_to_pts
from config import INPUT_SIZE, DEVICE


class FPSCounter:
    def __init__(self, window_size=15):
        self.times = deque(maxlen=window_size)

    def update(self, dur): self.times.append(dur)

    def get_fps(self): return 1.0 / (sum(self.times) / len(self.times)) if self.times else 0.0


class AIResultBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._pts_L, self._pts_R, self._age = None, None, 0

    def write(self, pts_L, pts_R):
        with self._lock: self._pts_L, self._pts_R, self._age = pts_L, pts_R, 0

    def read(self):
        with self._lock:
            self._age += 1
            return self._pts_L, self._pts_R, self._age


class AIInferenceThread(threading.Thread):
    def __init__(self, model, buf: AIResultBuffer, fps: FPSCounter):
        super().__init__(daemon=True)
        self.model, self.buf, self.fps, self._lock = model, buf, fps, threading.Lock()
        self._fL, self._fR, self._idx, self._last_idx = None, None, -1, -1
        self._stop = threading.Event()

    def update_frames(self, fL, fR, idx):
        with self._lock: self._fL, self._fR, self._idx = fL.copy(), fR.copy(), idx

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            with self._lock:
                fL, fR, idx = self._fL, self._fR, self._idx
            if fL is None or idx == self._last_idx:
                time.sleep(0.005);
                continue

            t0 = time.time()
            try:
                batch = torch.tensor(
                    np.stack([preprocess_for_model(fL, INPUT_SIZE), preprocess_for_model(fR, INPUT_SIZE)])).to(DEVICE)
                with torch.no_grad():
                    hm = self.model(batch).cpu().numpy()
                self.buf.write(decode_heatmap_to_pts(hm[0:1], fL.shape[0], fL.shape[1], fL),
                               decode_heatmap_to_pts(hm[1:2], fR.shape[0], fR.shape[1], fR))
                self._last_idx = idx
            except:
                pass
            self.fps.update(time.time() - t0)