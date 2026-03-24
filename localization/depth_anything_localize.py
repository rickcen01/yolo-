"""
Depth Anything V2 + YOLO 单目深度估计定位 | Monocular Depth Localization
=========================================================================
利用 NeurIPS 2024 最新成果 Depth Anything V2 从单张图片估计深度图，
结合 YOLO 检测气瓶位置，输出目标的相对深度及估算三维坐标。

特点：
  ✅ 只需一个摄像头，无需双目标定
  ✅ Depth Anything V2 为目前最强单目深度模型（7600+ Stars）
  ⚠️  默认输出相对深度（0~1），需用已知距离物体做比例标定才能得到毫米距离

使用方法：
  # 安装依赖（首次）
  pip install transformers timm ultralytics opencv-python

  # 运行（首次会自动下载模型 ~100MB）
  python depth_anything_localize.py
  python depth_anything_localize.py --source 0           # 摄像头实时
  python depth_anything_localize.py --source image.jpg   # 静态图片
  python depth_anything_localize.py --source video.mp4   # 视频文件
  python depth_anything_localize.py --metric             # 使用metric深度模型（室内/室外）
  python depth_anything_localize.py --calibrate 800      # 指定参考距离(mm)进行比例标定

模型说明：
  相对深度模型（默认）：Depth-Anything-V2-Small/Base/Large
  Metric深度模型：      Depth-Anything-V2-Metric-Indoor/Outdoor（直接输出米制距离）
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

# =====================================================================
# ⚙️  配置区域 - 根据实际情况修改
# =====================================================================

# YOLO 模型路径（优先使用自训练模型）
TRAINED_MODEL_PATH = "../gas_dataset/weights/best.pt"
DEFAULT_MODEL_PATH = "../yolo11x.pt"

# YOLO 检测置信度
CONF_THRESHOLD = 0.35

# 目标类别 ID（0=gas_cylinder；-1=检测所有类别）
TARGET_CLASS_ID = 0

# Depth Anything V2 模型规格：'vits'(小), 'vitb'(中), 'vitl'(大)
# 建议：实时用 'vits'，精度优先用 'vitl'
DEPTH_MODEL_SIZE = "vits"

# 推理图像尺寸（Depth Anything V2 输入尺寸，越大越精准但越慢）
DEPTH_INPUT_SIZE = 518

# 摄像头索引
CAMERA_INDEX = 0

# 图像分辨率
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# 比例标定：若已知某距离（mm）处的相对深度值，则可换算实际距离
# None = 不做换算，只显示相对深度
# 使用 --calibrate 参数或在此设置
SCALE_CALIBRATION_MM = None   # 例：1000.0 表示参考物在1000mm处

# 深度图颜色映射
DEPTH_COLORMAP = cv2.COLORMAP_INFERNO   # INFERNO / JET / PLASMA / MAGMA

# =====================================================================


# ─────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────

def load_depth_model(model_size: str, use_metric: bool = False, scene: str = "indoor"):
    """
    加载 Depth Anything V2 模型。

    优先尝试用 transformers pipeline（最简单）；
    若失败则尝试直接从 HuggingFace Hub 加载。

    Args:
        model_size: 'vits' | 'vitb' | 'vitl'
        use_metric: 是否使用 metric 深度模型（直接输出米制深度）
        scene:      metric 模型场景 'indoor' | 'outdoor'

    Returns:
        (model, processor, is_metric, device_str)
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_map = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
    size_label = size_map.get(model_size, "Small")

    if use_metric:
        scene_label = "Indoor" if scene == "indoor" else "Outdoor"
        model_id = f"depth-anything/Depth-Anything-V2-Metric-{scene_label}-{size_label}-hf"
        is_metric = True
    else:
        model_id = f"depth-anything/Depth-Anything-V2-{size_label}-hf"
        is_metric = False

    print(f"\n[Depth Anything V2] 加载模型: {model_id}")
    print(f"  设备: {device.upper()}  |  模式: {'Metric (米制)' if is_metric else '相对深度'}")
    size_mb_map = {'vits': 100, 'vitb': 400, 'vitl': 1400}
    print(f"  首次运行会自动下载模型（约 {size_mb_map.get(model_size, 100)}MB）...\n")

    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )
        print(f"[✓] 模型加载成功（via transformers pipeline）")
        return pipe, None, is_metric, device
    except Exception as e:
        print(f"[!] transformers pipeline 加载失败: {e}")
        print("[!] 尝试备用加载方式...")

    # 备用：直接用 AutoModel
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch

        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model = model.to(device).eval()
        print(f"[✓] 模型加载成功（via AutoModel）")
        return model, processor, is_metric, device
    except Exception as e2:
        print(f"[✗] 模型加载失败: {e2}")
        print("\n解决方法：")
        print("  1. 安装依赖: pip install transformers timm huggingface-hub")
        print("  2. 设置镜像（国内）: $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("  3. 手动下载模型后指定本地路径")
        sys.exit(1)


def load_yolo_model(model_path: str):
    """加载 YOLO 模型，自动选择最优可用模型"""
    from ultralytics import YOLO

    for path in [TRAINED_MODEL_PATH, model_path, "../yolo11n.pt"]:
        if path and os.path.exists(path):
            print(f"[✓] YOLO 模型: {path}")
            return YOLO(path)

    # 最后尝试直接下载
    print(f"[!] 本地模型不存在，尝试加载: {model_path}")
    try:
        return YOLO(os.path.basename(model_path))
    except Exception as e:
        print(f"[✗] YOLO 加载失败: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# 深度推理
# ─────────────────────────────────────────────────────────────────────

def infer_depth(model_or_pipe, processor, frame_bgr: np.ndarray,
                is_pipeline: bool, is_metric: bool, device: str) -> np.ndarray:
    """
    对一帧图像进行深度估计。

    Returns:
        depth_map: float32 numpy array, shape (H, W)
                   相对深度时值域 ~[0,1]（归一化）
                   metric 深度时单位为米
    """
    import torch
    from PIL import Image

    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    if is_pipeline:
        # transformers pipeline 接口
        result = model_or_pipe(pil_img)
        # pipeline 返回 PIL Image 或 tensor
        depth_raw = result["predicted_depth"]
        if hasattr(depth_raw, "numpy"):
            depth_np = depth_raw.squeeze().numpy()
        else:
            depth_np = np.array(depth_raw).squeeze()
    else:
        # AutoModel 接口
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_or_pipe(**inputs)
        depth_np = outputs.predicted_depth.squeeze().cpu().numpy()

    # resize 到原图尺寸
    h, w = frame_bgr.shape[:2]
    depth_resized = cv2.resize(
        depth_np.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    return depth_resized


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """将深度图归一化到 [0, 1]，用于可视化"""
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max - d_min < 1e-6:
        return np.zeros_like(depth_map)
    return (depth_map - d_min) / (d_max - d_min)


def depth_to_colormap(depth_map: np.ndarray,
                      colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """深度图转彩色可视化图（uint8 BGR）"""
    norm = normalize_depth(depth_map)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


# ─────────────────────────────────────────────────────────────────────
# 深度提取与坐标估算
# ─────────────────────────────────────────────────────────────────────

def extract_roi_depth(depth_map: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int,
                      method: str = "median") -> float | None:
    """
    提取检测框区域内的代表性深度值。

    method:
        'median'  - 中位数深度（最鲁棒，推荐）
        'center'  - 中心点深度
        'min'     - 最小深度（最近点，适合抓取）
        'mean'    - 均值（受噪声影响较大）
    """
    h, w = depth_map.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    if x2c <= x1c or y2c <= y1c:
        return None

    roi = depth_map[y1c:y2c, x1c:x2c]

    if method == "center":
        cy = (y1c + y2c) // 2 - y1c
        cx = (x1c + x2c) // 2 - x1c
        val = float(roi[cy, cx])
        return val if val > 1e-6 else None

    valid = roi[roi > 1e-6]
    if valid.size == 0:
        return None

    if method == "median":
        return float(np.median(valid))
    elif method == "min":
        return float(np.min(valid))
    elif method == "mean":
        return float(np.mean(valid))
    else:
        return float(np.median(valid))


def relative_to_mm(rel_depth: float, scale_mm: float) -> float:
    """
    将相对深度值换算为毫米距离。
    scale_mm: 参考物的真实距离（mm），对应其相对深度值 rel_depth。

    注意：Depth Anything V2 的相对深度值越大表示越近（disparity-like）。
    因此 Z_mm ≈ scale_mm × ref_rel_depth / rel_depth

    在未做精确比例标定时，结果仅供参考。
    """
    if rel_depth < 1e-6:
        return 0.0
    return scale_mm


def estimate_3d_from_depth(depth_rel: float, pixel_u: int, pixel_v: int,
                            frame_w: int, frame_h: int,
                            fov_deg: float = 60.0,
                            scale_ref: tuple | None = None) -> dict:
    """
    根据相对深度和像素位置估算3D坐标（相机坐标系）。

    原理：
      假设已知水平FOV（默认60°，可通过标定获取）
      fx ≈ W / (2 * tan(FOV/2))
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy

    Args:
        depth_rel:   目标的相对深度值
        pixel_u/v:   目标像素坐标（检测框中心）
        frame_w/h:   图像宽高
        fov_deg:     相机水平视场角（度）
        scale_ref:   (ref_rel_depth, ref_mm) 用于比例标定

    Returns:
        dict 包含：
            'depth_rel'  : 相对深度值
            'depth_mm'   : 估算毫米距离（若无标定则为 None）
            'X_mm'       : 水平偏移 mm（若无标定则为相对值）
            'Y_mm'       : 垂直偏移 mm
            'angle_x_deg': 水平偏角（度）
            'angle_y_deg': 垂直偏角（度）
    """
    cx_img = frame_w / 2.0
    cy_img = frame_h / 2.0

    # 估算焦距（像素）
    fov_rad = np.deg2rad(fov_deg)
    fx = frame_w / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # 假设像素为方形

    # 偏角
    angle_x = np.degrees(np.arctan2(pixel_u - cx_img, fx))
    angle_y = np.degrees(np.arctan2(pixel_v - cy_img, fy))

    # 若有比例标定，换算毫米距离
    depth_mm = None
    X_mm = None
    Y_mm = None

    if scale_ref is not None:
        ref_rel, ref_mm = scale_ref
        if ref_rel > 1e-6 and depth_rel > 1e-6:
            # Depth Anything V2 输出类似视差（越大越近）
            depth_mm = ref_mm * ref_rel / depth_rel
            X_mm = (pixel_u - cx_img) * depth_mm / fx
            Y_mm = (pixel_v - cy_img) * depth_mm / fy

    return {
        "depth_rel":   depth_rel,
        "depth_mm":    depth_mm,
        "X_mm":        X_mm,
        "Y_mm":        Y_mm,
        "angle_x_deg": angle_x,
        "angle_y_deg": angle_y,
    }


# ─────────────────────────────────────────────────────────────────────
# 比例标定工具
# ─────────────────────────────────────────────────────────────────────

class DepthScaleCalibrator:
    """
    交互式比例标定工具。
    用于将 Depth Anything V2 的相对深度换算为真实距离。

    使用方法：
      1. 将已知距离的物体（如气瓶）放在摄像头前方指定距离处
      2. 在程序运行时按 [c] 键，点击该物体区域
      3. 程序记录当前的相对深度值和对应的真实距离
      4. 后续所有帧均可换算为毫米距离
    """

    def __init__(self, ref_distance_mm: float = 1000.0):
        self.ref_distance_mm = ref_distance_mm  # 已知真实距离
        self.ref_rel_depth   = None             # 对应的相对深度值
        self.is_calibrated   = False
        self._click_point    = None
        self._waiting_click  = False

    def start_calibration(self, distance_mm: float = None):
        """开始标定：等待用户点击目标"""
        if distance_mm:
            self.ref_distance_mm = distance_mm
        self._waiting_click = True
        self._click_point   = None
        print(f"\n[标定] 请在图像窗口中点击已知距离 ({self.ref_distance_mm:.0f}mm) 的目标中心...")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调，记录点击位置"""
        if event == cv2.EVENT_LBUTTONDOWN and self._waiting_click:
            self._click_point   = (x, y)
            self._waiting_click = False

    def update_with_depth(self, depth_map: np.ndarray) -> bool:
        """
        若用户已点击，从深度图中读取点击位置的深度并完成标定。
        返回：是否刚完成标定
        """
        if self._click_point is None:
            return False

        x, y = self._click_point
        h, w = depth_map.shape[:2]
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        # 取点击位置 5×5 邻域的中位数深度
        x1, y1 = max(0, x - 2), max(0, y - 2)
        x2, y2 = min(w, x + 3), min(h, y + 3)
        patch   = depth_map[y1:y2, x1:x2]
        rel_val = float(np.median(patch))

        if rel_val < 1e-6:
            print(f"[标定] 点击位置深度无效（值={rel_val:.4f}），请重试。")
            self._click_point = None
            self._waiting_click = True
            return False

        self.ref_rel_depth = rel_val
        self.is_calibrated = True
        self._click_point  = None

        print(f"[标定完成] 参考点深度值={rel_val:.4f} ↔ 真实距离={self.ref_distance_mm:.0f}mm")
        print(f"  换算公式：Z_mm = {self.ref_distance_mm:.0f} × {rel_val:.4f} / depth_rel")
        return True

    @property
    def scale_ref(self) -> tuple | None:
        if self.is_calibrated and self.ref_rel_depth:
            return (self.ref_rel_depth, self.ref_distance_mm)
        return None

    def status_text(self) -> str:
        if self.is_calibrated:
            return f"已标定 ({self.ref_distance_mm:.0f}mm @ rel={self.ref_rel_depth:.3f})"
        elif self._waiting_click:
            return "等待点击目标..."
        else:
            return "未标定（按[c]键开始）"


# ─────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────

def draw_detections_with_depth(frame: np.ndarray,
                                depth_map: np.ndarray,
                                detections: list,
                                calibrator: DepthScaleCalibrator,
                                depth_method: str = "median") -> np.ndarray:
    """
    在原图上绘制 YOLO 检测框 + 深度信息。

    detections: ultralytics Results.boxes 列表
    """
    overlay = frame.copy()
    h, w    = frame.shape[:2]

    for box in detections:
        cls_id   = int(box.cls[0])
        conf_val = float(box.conf[0])

        if TARGET_CLASS_ID >= 0 and cls_id != TARGET_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 提取ROI深度
        depth_rel = extract_roi_depth(depth_map, x1, y1, x2, y2, method=depth_method)

        # 估算3D坐标
        coord3d = None
        if depth_rel is not None:
            coord3d = estimate_3d_from_depth(
                depth_rel, cx, cy, w, h,
                scale_ref=calibrator.scale_ref
            )

        # ── 绘制检测框 ──
        box_color = (0, 220, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

        # ── 绘制深度渐变条（右侧）──
        if depth_rel is not None:
            norm_val = normalize_depth(depth_map)[cy, cx]   # 0=近，1=远（归一化后）
            # Depth Anything V2 输出越大越近，所以用 1-norm
            near_far = 1.0 - norm_val
            bar_color = (
                int(255 * near_far),
                0,
                int(255 * (1 - near_far)),
            )
            bar_h    = max(4, int((y2 - y1) * (1 - near_far)))
            bar_x    = x2 + 5
            cv2.rectangle(overlay, (bar_x, y2 - bar_h), (bar_x + 8, y2), bar_color, -1)
            cv2.rectangle(overlay, (bar_x, y1),          (bar_x + 8, y2), (180, 180, 180), 1)

        # ── 文字标注 ──
        label_header = f"gas_cylinder  {conf_val:.2f}"
        cv2.putText(overlay, label_header,
                    (x1, max(12, y1 - 38)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        if depth_rel is not None:
            depth_text = f"深度(相对): {depth_rel:.4f}"
            cv2.putText(overlay, depth_text,
                        (x1, max(12, y1 - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

        if coord3d and coord3d["depth_mm"] is not None:
            Z  = coord3d["depth_mm"]
            X  = coord3d["X_mm"]
            Y  = coord3d["Y_mm"]
            ax = coord3d["angle_x_deg"]
            ay = coord3d["angle_y_deg"]
            coord_text = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"
            angle_text = f"偏角 H:{ax:+.1f}° V:{ay:+.1f}°"
            cv2.putText(overlay, coord_text,
                        (x1, max(12, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
            cv2.putText(overlay, angle_text,
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 0), 1)
        elif coord3d:
            ax = coord3d["angle_x_deg"]
            ay = coord3d["angle_y_deg"]
            angle_text = f"偏角 H:{ax:+.1f}° V:{ay:+.1f}° | 按[c]标定距离"
            cv2.putText(overlay, angle_text,
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 200, 100), 1)

    return overlay


def build_side_panel(depth_map: np.ndarray,
                     depth_colormap: int,
                     frame_h: int, panel_w: int = 320) -> np.ndarray:
    """构建右侧深度图侧边栏"""
    dcolor = depth_to_colormap(depth_map, depth_colormap)
    panel  = cv2.resize(dcolor, (panel_w, frame_h))

    # 添加说明文字
    cv2.putText(panel, "Depth Map", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(panel, "(Depth Anything V2)", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # 颜色条图例
    legend_x, legend_y = 10, frame_h - 80
    for i in range(100):
        ratio = i / 100.0
        gray  = int(ratio * 255)
        col   = cv2.applyColorMap(
            np.array([[gray]], dtype=np.uint8), depth_colormap
        )[0, 0].tolist()
        cv2.line(panel,
                 (legend_x + i * 3, legend_y),
                 (legend_x + i * 3, legend_y + 15),
                 col, 3)
    cv2.putText(panel, "近", (legend_x, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(panel, "远", (legend_x + 295, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return panel


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────

def run_on_source(source, args):
    """
    通用入口：支持摄像头（int）、图片（str）、视频（str）。
    """
    # ── 加载模型 ──────────────────────────────────────────────────────
    depth_model, depth_processor, is_metric, device = load_depth_model(
        model_size=args.depth_size,
        use_metric=args.metric,
        scene=args.scene,
    )
    is_pipeline = (depth_processor is None)

    yolo_model  = load_yolo_model(args.model)
    class_names = yolo_model.names
    print(f"[✓] YOLO 类别: {class_names}")

    # ── 标定器 ────────────────────────────────────────────────────────
    calibrator = DepthScaleCalibrator(
        ref_distance_mm=args.calibrate if args.calibrate else 1000.0
    )
    if args.calibrate:
        print(f"[标定] 已设置参考距离 {args.calibrate}mm，程序启动后按 [c] 点击目标。")

    # ── 打开视频源 ────────────────────────────────────────────────────
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_image = False
    elif source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        is_image = True
        cap = None
    else:
        cap = cv2.VideoCapture(source)
        is_image = False

    if not is_image and (cap is None or not cap.isOpened()):
        print(f"[错误] 无法打开视频源: {source}")
        sys.exit(1)

    # ── 窗口设置 ──────────────────────────────────────────────────────
    window_name = "Depth Anything V2 + YOLO 三维定位"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, calibrator.mouse_callback)

    fps_timer  = time.time()
    fps_frames = 0
    fps_val    = 0.0

    print("\n  操作说明:")
    print("  [c]     - 开始比例标定（点击已知距离目标）")
    print("  [s]     - 截图保存")
    print("  [d]     - 切换深度图显示模式")
    print("  [ESC/q] - 退出\n")

    show_depth_panel = True
    depth_method     = args.depth_method

    def get_frame():
        if is_image:
            img = cv2.imread(source)
            if img is None:
                print(f"[错误] 无法读取图片: {source}")
                sys.exit(1)
            return True, img
        ret, frame = cap.read()
        return ret, frame

    while True:
        ret, frame = get_frame()
        if not ret:
            print("[完成] 视频播放结束。")
            break

        # ── 深度估计 ────────────────────────────────────────────────
        t0 = time.time()
        depth_map = infer_depth(
            depth_model, depth_processor, frame,
            is_pipeline, is_metric, device
        )
        depth_ms = (time.time() - t0) * 1000

        # ── 标定更新 ─────────────────────────────────────────────────
        calibrator.update_with_depth(depth_map)

        # ── YOLO 检测 ────────────────────────────────────────────────
        yolo_results = yolo_model(
            frame,
            conf=args.conf,
            verbose=False,
        )

        all_boxes = []
        for result in yolo_results:
            all_boxes.extend(result.boxes)

        # ── 控制台输出 ───────────────────────────────────────────────
        for box in all_boxes:
            cls_id = int(box.cls[0])
            if TARGET_CLASS_ID >= 0 and cls_id != TARGET_CLASS_ID:
                continue
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            depth_rel = extract_roi_depth(depth_map, x1, y1, x2, y2, method=depth_method)
            if depth_rel is not None:
                coord3d = estimate_3d_from_depth(
                    depth_rel, cx, cy, frame.shape[1], frame.shape[0],
                    scale_ref=calibrator.scale_ref
                )
                if coord3d["depth_mm"] is not None:
                    print(
                        f"[气瓶] conf={conf_val:.2f}  "
                        f"深度(相对)={depth_rel:.4f}  "
                        f"X={coord3d['X_mm']:+.0f}mm  "
                        f"Y={coord3d['Y_mm']:+.0f}mm  "
                        f"Z={coord3d['depth_mm']:.0f}mm  "
                        f"偏角H={coord3d['angle_x_deg']:+.1f}°"
                    )
                else:
                    print(
                        f"[气瓶] conf={conf_val:.2f}  "
                        f"深度(相对)={depth_rel:.4f}  "
                        f"偏角H={coord3d['angle_x_deg']:+.1f}°  "
                        f"（按[c]完成比例标定可获得mm距离）"
                    )

        # ── 可视化 ───────────────────────────────────────────────────
        if not args.no_display:
            # 检测框+深度叠加
            display = draw_detections_with_depth(
                frame, depth_map, all_boxes, calibrator, depth_method
            )

            # FPS 计算
            fps_frames += 1
            if time.time() - fps_timer >= 1.0:
                fps_val    = fps_frames / (time.time() - fps_timer)
                fps_frames = 0
                fps_timer  = time.time()

            # 状态栏
            cv2.putText(display, f"FPS:{fps_val:.1f}  Depth:{depth_ms:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(display, f"标定: {calibrator.status_text()}",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(display, "[c]标定 [d]深度图 [s]截图 [ESC]退出",
                        (10, display.shape[0] - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

            # 拼接深度图侧边栏
            if show_depth_panel:
                panel   = build_side_panel(depth_map, DEPTH_COLORMAP,
                                           display.shape[0], panel_w=300)
                display = np.hstack([display, panel])

            cv2.imshow(window_name, display)

        # 静态图片：显示后等待按键
        wait_ms = 0 if is_image else 1
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            ref_dist = args.calibrate if args.calibrate else 1000.0
            calibrator.start_calibration(ref_dist)
        elif key == ord('d'):
            show_depth_panel = not show_depth_panel
        elif key == ord('s'):
            fname = f"depth_snapshot_{int(time.time())}.jpg"
            save_frame = display if not args.no_display else frame
            cv2.imwrite(fname, save_frame)
            print(f"[截图] 已保存: {fname}")

        if is_image:
            # 静态图片显示完后等待
            print("按任意键退出...")
            cv2.waitKey(0)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\n[完成] 程序已退出。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 + YOLO 单目深度定位系统"
    )
    parser.add_argument(
        "--source", default=str(CAMERA_INDEX),
        help="输入源：摄像头索引(0/1/...)、图片路径或视频路径（默认: 0）"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help="YOLO 模型路径"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"YOLO 置信度阈值（默认: {CONF_THRESHOLD}）"
    )
    parser.add_argument(
        "--depth-size", type=str, default=DEPTH_MODEL_SIZE,
        choices=["vits", "vitb", "vitl"],
        help="Depth Anything V2 模型规格（默认: vits）"
    )
    parser.add_argument(
        "--metric", action="store_true",
        help="使用 Metric 深度模型（直接输出米制距离，无需比例标定）"
    )
    parser.add_argument(
        "--scene", type=str, default="indoor",
        choices=["indoor", "outdoor"],
        help="Metric 模型场景类型（默认: indoor）"
    )
    parser.add_argument(
        "--calibrate", type=float, default=None,
        help="比例标定参考距离（mm），例如 --calibrate 800 表示参考物在800mm处"
    )
    parser.add_argument(
        "--depth-method", type=str, default="median",
        choices=["center", "median", "min", "mean"],
        help="ROI 深度提取方法（默认: median）"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="无图形界面模式（仅命令行输出）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析 source（数字→摄像头，字符串→文件）
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print("\n" + "=" * 65)
    print("  🔭 Depth Anything V2 + YOLO 单目深度定位系统")
    print("  NeurIPS 2024 最强单目深度模型 × YOLO 气瓶检测")
    print("=" * 65)

    try:
        run_on_source(source, args)
    except KeyboardInterrupt:
        print("\n[中断] 用户按下 Ctrl+C，程序退出。")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
