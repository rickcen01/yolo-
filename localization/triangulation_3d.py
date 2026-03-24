"""
多摄像头三角测量三维定位 | Multi-Camera Triangulation 3D Localization
======================================================================
基于极线几何（Epipolar Geometry）和 RANSAC 三角化，利用两路（或多路）
摄像头对同一目标进行精确三维定位。

核心技术栈：
  - YOLO11  ：目标检测（提取气瓶像素坐标）
  - OpenCV   ：极线约束 / 立体校正 / DLT 三角化
  - RANSAC   ：稳健三角化，抑制噪声
  - Matplotlib：实时 3D 轨迹可视化

两种运行模式：
  [A] 已做过双目标定（推荐）：直接读取 stereo_params.npz
  [B] 未标定（仅有基础矩阵 F）：使用本征矩阵恢复相对位姿

使用方法：
  # 已标定模式（精度更高）
  python triangulation_3d.py --mode calibrated

  # 未标定模式（仅需两台摄像头，无需棋盘格）
  python triangulation_3d.py --mode uncalibrated

  # 静态图片测试
  python triangulation_3d.py --left left.jpg --right right.jpg --mode calibrated

  # 调试：同时显示极线
  python triangulation_3d.py --show-epilines

依赖：
  opencv-contrib-python ultralytics numpy matplotlib scipy

参考：
  labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking (MIT)
  Hartley & Zisserman "Multiple View Geometry in Computer Vision"
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import warnings
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# =====================================================================
# ⚙️  全局配置
# =====================================================================

# ── 摄像头 ──────────────────────────────────────────────────────────
LEFT_CAM_INDEX   = 0      # 左/主摄像头
RIGHT_CAM_INDEX  = 1      # 右/副摄像头
FRAME_WIDTH      = 640
FRAME_HEIGHT     = 480

# ── YOLO ────────────────────────────────────────────────────────────
TRAINED_MODEL_PATH = "../gas_dataset/weights/best.pt"   # 自训练模型
DEFAULT_MODEL_PATH = "../yolo11x.pt"                    # 通用模型
CONF_THRESHOLD     = 0.40
TARGET_CLASS_ID    = 0     # 0=gas_cylinder；-1=所有类别

# ── 标定文件 ─────────────────────────────────────────────────────────
STEREO_PARAMS_FILE = "stereo_params.npz"   # 由 stereo_calibration.py 生成

# ── 三角测量 ─────────────────────────────────────────────────────────
# 重投影误差阈值（像素），超过此值的候选点将被 RANSAC 剔除
RANSAC_REPROJ_THRESH = 3.0

# 深度有效范围（mm）
MIN_Z_MM = 50.0
MAX_Z_MM = 8000.0

# 坐标平滑窗口（帧数）
SMOOTH_WINDOW = 7

# ── 可视化 ────────────────────────────────────────────────────────────
# 是否启动实时 3D 轨迹图（需要 matplotlib）
ENABLE_3D_PLOT = True

# JSON 坐标导出文件（None = 不导出）
EXPORT_JSON_PATH: Optional[str] = "detections_3d.json"

# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────

def load_stereo_params(params_file: str) -> dict:
    """
    从 .npz 文件加载双目标定参数。
    返回参数字典，包含相机内参、畸变、R/T/Q 等。
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"找不到标定文件：{params_file}\n"
            "请先运行 stereo_calibration.py 完成双目标定，\n"
            "或使用 --mode uncalibrated 进入无标定模式。"
        )
    data = np.load(params_file, allow_pickle=True)
    params = {k: data[k] for k in data.files}
    print(f"[✓] 标定参数已加载: {params_file}")
    print(f"    基线={float(params['baseline_mm']):.2f}mm  "
          f"焦距={float(params['focal_px']):.2f}px")
    return params


def build_projection_matrices(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    从标定参数构建左/右相机投影矩阵 P1, P2（校正后）。

    已标定且做过 stereoRectify 的情况下，P1/P2 已由标定程序给出；
    此处直接取出即可。
    """
    P1 = params["P1"].astype(np.float64)
    P2 = params["P2"].astype(np.float64)
    return P1, P2


def build_rectify_maps(params: dict, image_size: tuple) -> tuple:
    """
    生成左/右相机的畸变校正 + 极线对齐映射表。
    返回 (map1_l, map2_l, map1_r, map2_r)
    """
    mtx_l  = params["mtx_left"].astype(np.float64)
    dist_l = params["dist_left"].astype(np.float64)
    mtx_r  = params["mtx_right"].astype(np.float64)
    dist_r = params["dist_right"].astype(np.float64)
    R1, R2 = params["R1"], params["R2"]
    P1, P2 = params["P1"], params["P2"]

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )
    return map1_l, map2_l, map1_r, map2_r


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray,
                           pt_left: np.ndarray, pt_right: np.ndarray) -> np.ndarray:
    """
    直接线性变换（DLT）三角化单对对应点。

    Args:
        P1, P2  : (3×4) 投影矩阵
        pt_left : (2,) 左图像素坐标 [u, v]
        pt_right: (2,) 右图像素坐标 [u, v]

    Returns:
        X_world: (3,) 三维坐标 (X, Y, Z)，单位与标定时一致（mm）
    """
    pts_l = pt_left.reshape(1, 1, 2).astype(np.float32)
    pts_r = pt_right.reshape(1, 1, 2).astype(np.float32)

    X_hom = cv2.triangulatePoints(P1, P2, pts_l, pts_r)  # (4, 1)
    X_hom = X_hom[:, 0]

    if abs(X_hom[3]) < 1e-9:
        return None

    X = X_hom[:3] / X_hom[3]
    return X.astype(np.float64)


def triangulate_robust(P1: np.ndarray, P2: np.ndarray,
                        pts_left: np.ndarray, pts_right: np.ndarray,
                        reproj_thresh: float = RANSAC_REPROJ_THRESH
                        ) -> tuple[np.ndarray | None, float]:
    """
    RANSAC 稳健三角化：对同一目标的多个候选像素点集进行三角化，
    选出重投影误差最小的结果。

    Args:
        P1, P2     : (3×4) 投影矩阵
        pts_left   : (N, 2) 左图候选点（如检测框内的采样点）
        pts_right  : (N, 2) 右图对应候选点
        reproj_thresh: 重投影误差阈值（像素）

    Returns:
        (best_X, best_reproj_error)  或  (None, inf)
    """
    if len(pts_left) == 0 or len(pts_right) == 0:
        return None, float("inf")

    best_X     = None
    best_err   = float("inf")
    best_inliers = 0

    for pl, pr in zip(pts_left, pts_right):
        X = triangulate_point_dlt(P1, P2, pl, pr)
        if X is None:
            continue

        # 检查深度是否有效
        Z = X[2]
        if not (MIN_Z_MM < Z < MAX_Z_MM):
            continue

        # 计算重投影误差
        def reproj_err(P, pt2d, X3d):
            Xh = np.append(X3d, 1.0)
            p  = P @ Xh
            if abs(p[2]) < 1e-9:
                return float("inf")
            p_norm = p[:2] / p[2]
            return float(np.linalg.norm(p_norm - pt2d))

        err_l = reproj_err(P1, pl, X)
        err_r = reproj_err(P2, pr, X)
        total_err = (err_l + err_r) / 2.0

        if total_err < best_err:
            best_err = total_err
            best_X   = X

    if best_X is not None and best_err <= reproj_thresh * 2:
        return best_X, best_err
    return None, float("inf")


def sample_roi_points(x1: int, y1: int, x2: int, y2: int,
                      n_samples: int = 9) -> np.ndarray:
    """
    在检测框内均匀采样 n_samples 个像素点，
    包括中心、四角、四边中点，用于 RANSAC 三角化。
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1

    points = [
        [cx,         cy],              # 中心
        [cx,         cy - h * 0.25],   # 上
        [cx,         cy + h * 0.25],   # 下
        [cx - w * 0.25, cy],           # 左
        [cx + w * 0.25, cy],           # 右
        [cx - w * 0.3,  cy - h * 0.3], # 左上
        [cx + w * 0.3,  cy - h * 0.3], # 右上
        [cx - w * 0.3,  cy + h * 0.3], # 左下
        [cx + w * 0.3,  cy + h * 0.3], # 右下
    ]
    return np.array(points[:n_samples], dtype=np.float32)


def compute_fundamental_matrix(pts_left: np.ndarray,
                                pts_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 RANSAC 从对应点对计算基础矩阵 F。
    适用于无标定模式（仅需匹配点对）。

    Returns:
        (F, mask)
    """
    F, mask = cv2.findFundamentalMat(
        pts_left, pts_right,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999,
    )
    return F, mask


def epiline_distance(F: np.ndarray,
                     pt_left: np.ndarray,
                     pt_right: np.ndarray) -> float:
    """
    计算右图中点 pt_right 到其极线（由 pt_left 经 F 确定）的距离。
    用于极线约束匹配质量评估。

    Returns:
        距离（像素），越小表示匹配越好
    """
    pt_l_h = np.array([pt_left[0], pt_left[1], 1.0])
    line   = F @ pt_l_h   # 极线系数 [a, b, c]（ax+by+c=0）

    a, b, c = line
    norm = np.sqrt(a**2 + b**2)
    if norm < 1e-9:
        return float("inf")

    dist = abs(a * pt_right[0] + b * pt_right[1] + c) / norm
    return float(dist)


def draw_epilines_on_pair(img_left: np.ndarray, img_right: np.ndarray,
                          pts_left: np.ndarray, pts_right: np.ndarray,
                          F: np.ndarray, n_lines: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    在左右图像上绘制极线，用于可视化验证标定质量。
    """
    vis_l = img_left.copy()
    vis_r = img_right.copy()
    h, w  = vis_l.shape[:2]

    n = min(n_lines, len(pts_left))
    colors = [
        tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)
    ]

    # 左图点 → 右图极线
    lines_r = cv2.computeCorrespondEpilines(
        pts_left[:n].reshape(-1, 1, 2), 1, F
    ).reshape(-1, 3)

    # 右图点 → 左图极线
    lines_l = cv2.computeCorrespondEpilines(
        pts_right[:n].reshape(-1, 1, 2), 2, F
    ).reshape(-1, 3)

    for i in range(n):
        col = colors[i]
        # 右图极线
        a, b, c = lines_r[i]
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-9 else 0
        x1_, y1_ = w, int((-c - a * w) / b) if abs(b) > 1e-9 else h
        cv2.line(vis_r, (x0, y0), (x1_, y1_), col, 1)
        cv2.circle(vis_r, tuple(map(int, pts_right[i])), 5, col, -1)

        # 左图极线
        a, b, c = lines_l[i]
        x0, y0 = 0, int(-c / b) if abs(b) > 1e-9 else 0
        x1_, y1_ = w, int((-c - a * w) / b) if abs(b) > 1e-9 else h
        cv2.line(vis_l, (x0, y0), (x1_, y1_), col, 1)
        cv2.circle(vis_l, tuple(map(int, pts_left[i])), 5, col, -1)

    return vis_l, vis_r


# ─────────────────────────────────────────────────────────────────────
# 跨视角目标匹配
# ─────────────────────────────────────────────────────────────────────

def match_detections_across_views(
    dets_left: list[dict],
    dets_right: list[dict],
    F: np.ndarray,
    epiline_thresh: float = 15.0
) -> list[tuple[dict, dict, float]]:
    """
    利用极线约束在两视角的检测结果之间建立对应关系。

    策略：
      对于左图中的每个检测框，在右图中寻找满足以下条件的对应框：
        1. 类别相同
        2. 检测框中心点的极线距离 < epiline_thresh（像素）
        3. 在所有满足条件的候选中，选极线距离最小的

    Args:
        dets_left / dets_right: 检测结果列表，每项包含：
            {'bbox': (x1,y1,x2,y2), 'cls_id': int, 'conf': float,
             'cx': float, 'cy': float}
        F              : 基础矩阵
        epiline_thresh : 极线距离容差（像素）

    Returns:
        matched: list of (det_left, det_right, epiline_dist)
                 已按 epiline_dist 从小到大排序
    """
    matched = []
    used_right = set()

    for i, det_l in enumerate(dets_left):
        pt_l  = np.array([[det_l["cx"], det_l["cy"]]], dtype=np.float32)
        best_j    = -1
        best_dist = float("inf")

        for j, det_r in enumerate(dets_right):
            if j in used_right:
                continue
            if det_r["cls_id"] != det_l["cls_id"]:
                continue

            pt_r   = np.array([det_r["cx"], det_r["cy"]], dtype=np.float32)
            edist  = epiline_distance(F, pt_l[0], pt_r)

            if edist < epiline_thresh and edist < best_dist:
                best_dist = edist
                best_j    = j

        if best_j >= 0:
            matched.append((det_l, dets_right[best_j], best_dist))
            used_right.add(best_j)

    matched.sort(key=lambda x: x[2])
    return matched


# ─────────────────────────────────────────────────────────────────────
# 坐标平滑与历史记录
# ─────────────────────────────────────────────────────────────────────

class SmoothCoordTracker:
    """
    对同一目标的 3D 坐标进行时序平滑，抑制帧间抖动。
    使用滑动窗口中位数滤波。
    """

    def __init__(self, window: int = SMOOTH_WINDOW):
        self.window  = window
        self.history: collections.deque = collections.deque(maxlen=window)
        self.last_update = time.time()

    def update(self, xyz: np.ndarray) -> np.ndarray:
        """加入新坐标，返回平滑后的坐标"""
        self.history.append(xyz.copy())
        self.last_update = time.time()
        if len(self.history) == 1:
            return xyz.copy()
        hist = np.stack(list(self.history), axis=0)  # (N, 3)
        return np.median(hist, axis=0)

    def get_smooth(self) -> np.ndarray | None:
        if not self.history:
            return None
        hist = np.stack(list(self.history), axis=0)
        return np.median(hist, axis=0)

    def is_stale(self, timeout: float = 1.0) -> bool:
        return (time.time() - self.last_update) > timeout


# ─────────────────────────────────────────────────────────────────────
# 实时 3D 可视化（Matplotlib）
# ─────────────────────────────────────────────────────────────────────

class Live3DPlotter:
    """
    实时 3D 散点图，显示目标在三维空间中的轨迹。
    在后台线程中异步更新，不阻塞主循环。
    """

    def __init__(self, max_trail: int = 80):
        self.max_trail = max_trail
        self._trail: collections.deque = collections.deque(maxlen=max_trail)
        self._fig    = None
        self._ax     = None
        self._enabled = False
        self._init_plot()

    def _init_plot(self):
        try:
            import matplotlib
            matplotlib.use("TkAgg")   # 避免主线程冲突；失败时降级
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            self._fig = plt.figure("3D Object Trajectory", figsize=(6, 5))
            self._ax  = self._fig.add_subplot(111, projection="3d")
            self._ax.set_xlabel("X (mm)")
            self._ax.set_ylabel("Z (mm)")
            self._ax.set_zlabel("Y (mm)")
            self._ax.set_title("气瓶 3D 位置轨迹")
            plt.ion()
            plt.show(block=False)
            self._plt = plt
            self._enabled = True
            print("[✓] 3D 轨迹图已启动")
        except Exception as e:
            print(f"[!] 3D 图初始化失败（{e}），将跳过可视化。")
            self._enabled = False

    def add_point(self, X: float, Y: float, Z: float):
        """添加新轨迹点"""
        self._trail.append((X, Y, Z))

    def refresh(self):
        """刷新绘图（应在主线程中调用）"""
        if not self._enabled or not self._trail:
            return
        try:
            xs, ys, zs = zip(*self._trail)
            self._ax.cla()
            self._ax.set_xlabel("X (mm)")
            self._ax.set_ylabel("Z (mm)")
            self._ax.set_zlabel("Y (mm)")
            self._ax.set_title("气瓶 3D 位置轨迹")

            # 历史轨迹（渐变透明）
            n = len(xs)
            alphas = np.linspace(0.15, 1.0, n)
            for i in range(n - 1):
                self._ax.plot(
                    [xs[i], xs[i+1]],
                    [zs[i], zs[i+1]],
                    [ys[i], ys[i+1]],
                    color=(0.2, 0.6, 1.0, alphas[i]),
                    linewidth=1.2,
                )

            # 当前点（大红点）
            self._ax.scatter(
                [xs[-1]], [zs[-1]], [ys[-1]],
                c="red", s=80, zorder=5, label="当前位置"
            )

            # 相机原点
            self._ax.scatter([0], [0], [0], c="green", s=60,
                             marker="^", label="相机原点")

            self._ax.legend(loc="upper left", fontsize=8)
            self._plt.tight_layout()
            self._plt.pause(0.001)
        except Exception:
            pass  # 绘图异常不影响主流程


# ─────────────────────────────────────────────────────────────────────
# JSON 导出
# ─────────────────────────────────────────────────────────────────────

class CoordExporter:
    """将每帧检测到的 3D 坐标追加写入 JSON Lines 文件"""

    def __init__(self, filepath: Optional[str]):
        self.filepath = filepath
        self._records: list = []
        if filepath:
            # 清空旧文件
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("")
            print(f"[✓] 坐标将导出至: {filepath}")

    def record(self, frame_id: int, timestamp: float,
               detections_3d: list[dict]):
        """记录一帧的检测结果"""
        if not self.filepath:
            return
        entry = {
            "frame":     frame_id,
            "timestamp": round(timestamp, 4),
            "detections": [
                {
                    "class":   d.get("label", "unknown"),
                    "conf":    round(d.get("conf", 0), 4),
                    "X_mm":    round(float(d["xyz"][0]), 2),
                    "Y_mm":    round(float(d["xyz"][1]), 2),
                    "Z_mm":    round(float(d["xyz"][2]), 2),
                    "reproj_err": round(d.get("reproj_err", -1), 3),
                    "epiline_dist": round(d.get("epiline_dist", -1), 3),
                }
                for d in detections_3d if d.get("xyz") is not None
            ],
        }
        if entry["detections"]:
            self._records.append(entry)
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self):
        """打印导出摘要"""
        if self.filepath and self._records:
            n_dets = sum(len(r["detections"]) for r in self._records)
            print(f"\n[导出] 共记录 {len(self._records)} 帧，{n_dets} 次检测 → {self.filepath}")


# ─────────────────────────────────────────────────────────────────────
# 绘制函数
# ─────────────────────────────────────────────────────────────────────

def draw_detections_3d(frame: np.ndarray, results_3d: list[dict],
                       view_name: str = "LEFT") -> np.ndarray:
    """
    在单视角图像上绘制检测框和三维坐标标注。

    results_3d: list of dict，每项包含：
        bbox, label, conf, xyz (or None), reproj_err, epiline_dist
    """
    vis = frame.copy()
    for det in results_3d:
        x1, y1, x2, y2 = det["bbox"]
        label     = det.get("label", "obj")
        conf_val  = det.get("conf", 0.0)
        xyz       = det.get("xyz")
        rerr      = det.get("reproj_err", -1)
        edist     = det.get("epiline_dist", -1)

        # 颜色：有3D坐标=绿，无=橙
        color = (0, 220, 50) if xyz is not None else (0, 140, 255)

        # 检测框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

        # 标题行
        header = f"{label}  {conf_val:.2f}"
        cv2.putText(vis, header, (x1, max(14, y1 - 42)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        if xyz is not None:
            X, Y, Z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            coord = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"
            cv2.putText(vis, coord, (x1, max(14, y1 - 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            if rerr >= 0:
                qa_text = f"reproj={rerr:.1f}px  eline={edist:.1f}px"
                cv2.putText(vis, qa_text, (x1, max(14, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)
        else:
            cv2.putText(vis, "跨视角未匹配", (x1, max(14, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1)

    # 视角标签
    cv2.putText(vis, view_name, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    return vis


def draw_status_bar(combined: np.ndarray, fps: float, n_det: int,
                    n_match: int, frame_id: int) -> np.ndarray:
    """在拼接图底部绘制状态栏"""
    h, w = combined.shape[:2]
    bar_h = 26
    bar   = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    txt = (f"FPS:{fps:.1f}  检测:{n_det}  匹配:{n_match}  帧:{frame_id}"
           "  | [e]极线 [3]3D图 [s]截图 [ESC]退出")
    cv2.putText(bar, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.vstack([combined, bar])


# ─────────────────────────────────────────────────────────────────────
# 主运行循环
# ─────────────────────────────────────────────────────────────────────

def run_triangulation(args):
    """
    主循环：双目摄像头 + YOLO + 三角测量 → 实时 3D 坐标输出。
    """
    print("\n" + "=" * 65)
    print("  📐 多摄像头三角测量三维定位系统  v1.0")
    print("  YOLO11 + 极线几何 + RANSAC 三角化")
    print("=" * 65)

    # ── 加载标定参数 ─────────────────────────────────────────────────
    try:
        params = load_stereo_params(STEREO_PARAMS_FILE)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    P1, P2 = build_projection_matrices(params)
    F_mat  = params.get("F")
    if F_mat is not None:
        F_mat = F_mat.astype(np.float64)
        print(f"[✓] 已加载基础矩阵 F（从标定参数）")
    else:
        print("[!] 标定参数中未包含基础矩阵 F，将使用 P1/P2 估算极线约束。")

    image_size = (FRAME_WIDTH, FRAME_HEIGHT)
    map1_l, map2_l, map1_r, map2_r = build_rectify_maps(params, image_size)

    # ── 打开摄像头 ───────────────────────────────────────────────────
    is_static = False
    if args.left and args.right:
        # 静态图片模式
        frame_l_static = cv2.imread(args.left)
        frame_r_static = cv2.imread(args.right)
        if frame_l_static is None or frame_r_static is None:
            print(f"[错误] 无法读取图片: {args.left} / {args.right}")
            sys.exit(1)
        is_static = True
        cap_l = cap_r = None
        print(f"[静态模式] 左: {args.left}  右: {args.right}")
    else:
        cap_l = cv2.VideoCapture(LEFT_CAM_INDEX)
        cap_r = cv2.VideoCapture(RIGHT_CAM_INDEX)
        if not cap_l.isOpened():
            print(f"[错误] 无法打开左摄像头 (索引={LEFT_CAM_INDEX})")
            sys.exit(1)
        if not cap_r.isOpened():
            print(f"[错误] 无法打开右摄像头 (索引={RIGHT_CAM_INDEX})")
            sys.exit(1)
        for cap in [cap_l, cap_r]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── 加载 YOLO ────────────────────────────────────────────────────
    from ultralytics import YOLO
    yolo_model = None
    for path in [TRAINED_MODEL_PATH, DEFAULT_MODEL_PATH, "../yolo11n.pt"]:
        if os.path.exists(path):
            yolo_model = YOLO(path)
            print(f"[✓] YOLO 模型: {path}")
            break
    if yolo_model is None:
        print("[!] 尝试下载默认模型...")
        yolo_model = YOLO("yolo11n.pt")

    class_names = yolo_model.names

    # ── 辅助对象 ─────────────────────────────────────────────────────
    tracker    = SmoothCoordTracker(window=SMOOTH_WINDOW)
    exporter   = CoordExporter(EXPORT_JSON_PATH if not args.no_export else None)
    plotter    = Live3DPlotter() if ENABLE_3D_PLOT and not args.no_3dplot else None

    # ── 状态变量 ─────────────────────────────────────────────────────
    fps_timer   = time.time()
    fps_frames  = 0
    fps_val     = 0.0
    frame_id    = 0
    show_epilines = args.show_epilines

    print("\n  操作说明:")
    print("  [e]     - 切换极线显示")
    print("  [3]     - 切换3D轨迹图")
    print("  [s]     - 截图保存")
    print("  [ESC/q] - 退出\n")

    while True:
        # ── 读取帧 ───────────────────────────────────────────────────
        if is_static:
            frame_l_raw = frame_l_static.copy()
            frame_r_raw = frame_r_static.copy()
        else:
            ret_l, frame_l_raw = cap_l.read()
            ret_r, frame_r_raw = cap_r.read()
            if not ret_l or not ret_r:
                print("[警告] 摄像头读取失败，重试...")
                time.sleep(0.05)
                continue

        frame_id += 1

        # ── 立体校正 ─────────────────────────────────────────────────
        frame_l = cv2.remap(frame_l_raw, map1_l, map2_l, cv2.INTER_LINEAR)
        frame_r = cv2.remap(frame_r_raw, map1_r, map2_r, cv2.INTER_LINEAR)

        # ── YOLO 检测（左右同时检测）────────────────────────────────
        res_l = yolo_model(frame_l, conf=CONF_THRESHOLD, verbose=False)
        res_r = yolo_model(frame_r, conf=CONF_THRESHOLD, verbose=False)

        def parse_detections(results) -> list[dict]:
            dets = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if TARGET_CLASS_ID >= 0 and cls_id != TARGET_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    dets.append({
                        "bbox":   (x1, y1, x2, y2),
                        "cls_id": cls_id,
                        "conf":   float(box.conf[0]),
                        "label":  class_names.get(cls_id, str(cls_id)),
                        "cx":     cx,
                        "cy":     cy,
                    })
            return dets

        dets_l = parse_detections(res_l)
        dets_r = parse_detections(res_r)

        # ── 跨视角匹配 ───────────────────────────────────────────────
        matched_pairs = []
        results_3d_l  = list(dets_l)  # 默认全部左图检测框（含未匹配的）

        if F_mat is not None and dets_l and dets_r:
            matched_pairs = match_detections_across_views(
                dets_l, dets_r, F_mat, epiline_thresh=20.0
            )

        # ── 三角测量 ─────────────────────────────────────────────────
        det_records_3d = []

        for det_l, det_r, edist in matched_pairs:
            # 采样 ROI 候选点
            pts_l = sample_roi_points(*det_l["bbox"])
            pts_r = sample_roi_points(*det_r["bbox"])

            # RANSAC 三角化
            X3d, rerr = triangulate_robust(P1, P2, pts_l, pts_r)

            xyz_smooth = None
            if X3d is not None:
                xyz_smooth = tracker.update(X3d)

                det_record = {
                    **det_l,
                    "xyz":          xyz_smooth,
                    "reproj_err":   rerr,
                    "epiline_dist": edist,
                }
                det_records_3d.append(det_record)

                # 控制台输出
                X, Y, Z = float(xyz_smooth[0]), float(xyz_smooth[1]), float(xyz_smooth[2])
                print(
                    f"[{det_l['label']}] conf={det_l['conf']:.2f}  "
                    f"X={X:+7.1f}  Y={Y:+7.1f}  Z={Z:7.1f} mm  "
                    f"reproj={rerr:.1f}px  eline={edist:.1f}px"
                )

                # 3D 轨迹更新
                if plotter:
                    plotter.add_point(X, Y, Z)

        # ── JSON 导出 ────────────────────────────────────────────────
        exporter.record(frame_id, time.time(), det_records_3d)

        # ── 可视化 ───────────────────────────────────────────────────
        # 把三角化结果标注回左图
        vis_l = draw_detections_3d(frame_l, det_records_3d, view_name="LEFT")
        vis_r = draw_detections_3d(frame_r, [], view_name="RIGHT")

        # 在右图上标右侧检测框（未匹配的）
        for det_r_item in dets_r:
            x1, y1, x2, y2 = det_r_item["bbox"]
            cv2.rectangle(vis_r, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(vis_r, f"{det_r_item['label']} {det_r_item['conf']:.2f}",
                        (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1)

        # 极线可视化
        if show_epilines and F_mat is not None and dets_l and dets_r:
            pts_l_all = np.array([[d["cx"], d["cy"]] for d in dets_l], dtype=np.float32)
            pts_r_all = np.array([[d["cx"], d["cy"]] for d in dets_r], dtype=np.float32)
            if len(pts_l_all) > 0 and len(pts_r_all) > 0:
                vis_l, vis_r = draw_epilines_on_pair(vis_l, vis_r,
                                                      pts_l_all, pts_r_all, F_mat)

        # 添加极线对齐网格（校正质量参考）
        for y_ln in range(0, FRAME_HEIGHT, 50):
            cv2.line(vis_l, (0, y_ln), (FRAME_WIDTH, y_ln), (40, 40, 40), 1)
            cv2.line(vis_r, (0, y_ln), (FRAME_WIDTH, y_ln), (40, 40, 40), 1)

        # FPS 计算
        fps_frames += 1
        if time.time() - fps_timer >= 1.0:
            fps_val    = fps_frames / (time.time() - fps_timer)
            fps_frames = 0
            fps_timer  = time.time()

        combined = np.hstack([vis_l, vis_r])
        combined = draw_status_bar(
            combined, fps_val, len(dets_l) + len(dets_r),
            len(matched_pairs), frame_id
        )

        cv2.imshow("Multi-Camera Triangulation | 多摄像头三角测量", combined)

        # 定期刷新 3D 图
        if plotter and frame_id % 5 == 0:
            plotter.refresh()

        # ── 键盘交互 ─────────────────────────────────────────────────
        wait_ms = 0 if is_static else 1
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('e'):
            show_epilines = not show_epilines
            print(f"[极线] {'开启' if show_epilines else '关闭'}")
        elif key == ord('3') and plotter:
            plotter.refresh()
        elif key == ord('s'):
            fname = f"tri_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, combined)
            print(f"[截图] 已保存: {fname}")

        if is_static:
            print("按任意键退出...")
            cv2.waitKey(0)
            break

    # ── 清理 ─────────────────────────────────────────────────────────
    if cap_l:
        cap_l.release()
    if cap_r:
        cap_r.release()
    cv2.destroyAllWindows()
    exporter.summary()
    print("\n[完成] 程序已退出。")


# ─────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="多摄像头三角测量三维定位系统 | Multi-Camera Triangulation"
    )
    parser.add_argument(
        "--left", type=str, default=None,
        help="左图片路径（静态测试用）"
    )
    parser.add_argument(
        "--right", type=str, default=None,
        help="右图片路径（静态测试用）"
    )
    parser.add_argument(
        "--mode", type=str, default="calibrated",
        choices=["calibrated", "uncalibrated"],
        help="运行模式：calibrated（需要 stereo_params.npz）/ uncalibrated"
    )
    parser.add_argument(
        "--show-epilines", action="store_true",
        help="显示极线（用于验证标定质量）"
    )
    parser.add_argument(
        "--no-3dplot", action="store_true",
        help="禁用实时 3D 轨迹图（提升性能）"
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="禁用 JSON 坐标导出"
    )
    parser.add_argument(
        "--left-cam", type=int, default=LEFT_CAM_INDEX,
        help=f"左摄像头索引（默认: {LEFT_CAM_INDEX}）"
    )
    parser.add_argument(
        "--right-cam", type=int, default=RIGHT_CAM_INDEX,
        help=f"右摄像头索引（默认: {RIGHT_CAM_INDEX}）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    LEFT_CAM_INDEX  = args.left_cam
    RIGHT_CAM_INDEX = args.right_cam

    try:
        run_triangulation(args)
    except KeyboardInterrupt:
        print("\n[中断] 用户按下 Ctrl+C，程序退出。")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
