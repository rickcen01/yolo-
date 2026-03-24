"""
双目立体视觉 + YOLO 三维定位 | Stereo 3D Localization
======================================================
在 YOLO 检测到气瓶后，利用双目视差计算目标的真实三维坐标 (X, Y, Z)。

依赖：
  - stereo_params.npz   (由 stereo_calibration.py 生成)
  - stereo_maps.npz     (由 stereo_calibration.py 生成)
  - YOLO 模型文件（../yolo11x.pt 或自训练模型）

运行：
  python stereo_3d_localize.py
  python stereo_3d_localize.py --model ../best.pt --conf 0.5
  python stereo_3d_localize.py --no-display          # 无界面模式（纯输出）

输出坐标说明：
  X = 水平偏移（正 = 右偏）  单位：mm
  Y = 垂直偏移（正 = 下偏）  单位：mm
  Z = 深度（距相机距离）     单位：mm
"""

import argparse
import sys
import os
import time
import collections

import cv2
import numpy as np

# =====================================================================
# ⚙️  配置区域
# =====================================================================

# 摄像头索引
LEFT_CAM_INDEX  = 0
RIGHT_CAM_INDEX = 1

# YOLO 模型路径（优先使用自训练模型，否则回退到通用模型）
DEFAULT_MODEL_PATH = "../yolo11x.pt"
TRAINED_MODEL_PATH = "../gas_dataset/weights/best.pt"   # 训练后的模型

# 检测置信度阈值
CONF_THRESHOLD = 0.40

# 目标类别 ID（0 = gas_cylinder；-1 = 检测所有类别）
TARGET_CLASS_ID = 0

# SGBM 视差参数（可根据实际场景调整）
SGBM_NUM_DISPARITIES = 96    # 必须是 16 的倍数，增大可检测更近的物体
SGBM_BLOCK_SIZE      = 7     # 奇数，5~11
SGBM_MIN_DISPARITY   = 0
SGBM_UNIQUENESS      = 10
SGBM_SPECKLE_WIN     = 100
SGBM_SPECKLE_RANGE   = 32
SGBM_DISP12MAX       = 1
SGBM_P1_FACTOR       = 8     # P1 = P1_FACTOR * 3 * blockSize^2
SGBM_P2_FACTOR       = 32    # P2 = P2_FACTOR * 3 * blockSize^2

# 深度有效范围过滤（mm）
MIN_DEPTH_MM = 100.0
MAX_DEPTH_MM = 5000.0

# 坐标平滑滤波：使用最近 N 帧的中位数
SMOOTH_WINDOW = 5

# 图像分辨率
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# 标定参数文件路径
PARAMS_FILE = "stereo_params.npz"
MAPS_FILE   = "stereo_maps.npz"

# =====================================================================


def load_stereo_params(params_file, maps_file, image_size):
    """
    加载双目标定参数和校正映射表。
    若 maps_file 不存在，则从 params_file 重新生成映射表。
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(
            f"找不到标定参数文件：{params_file}\n"
            "请先运行 stereo_calibration.py 完成双目标定。"
        )

    data = np.load(params_file)
    params = {k: data[k] for k in data.files}

    # 尝试加载已保存的映射表（速度更快）
    if os.path.exists(maps_file):
        mdata = np.load(maps_file)
        map1_l = mdata["map1_l"]
        map2_l = mdata["map2_l"]
        map1_r = mdata["map1_r"]
        map2_r = mdata["map2_r"]
        print(f"[✓] 已加载校正映射表: {maps_file}")
    else:
        print(f"[!] 未找到 {maps_file}，重新生成映射表...")
        mtx_l = params["mtx_left"]
        dist_l = params["dist_left"]
        mtx_r = params["mtx_right"]
        dist_r = params["dist_right"]
        R1 = params["R1"]
        R2 = params["R2"]
        P1 = params["P1"]
        P2 = params["P2"]

        map1_l, map2_l = cv2.initUndistortRectifyMap(
            mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
        )
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
        )

    maps = {
        "map1_l": map1_l, "map2_l": map2_l,
        "map1_r": map1_r, "map2_r": map2_r,
    }

    print(f"[✓] 已加载标定参数: {params_file}")
    print(f"    基线: {float(params['baseline_mm']):.2f} mm  |  "
          f"焦距: {float(params['focal_px']):.2f} px")

    return params, maps


def build_sgbm_matcher():
    """
    构建 StereoSGBM 视差匹配器。
    SGBM（半全局匹配）比普通 BM 精度更高，适合室内场景。
    """
    bs = SGBM_BLOCK_SIZE
    P1 = SGBM_P1_FACTOR * 3 * bs * bs
    P2 = SGBM_P2_FACTOR * 3 * bs * bs

    matcher_left = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=bs,
        P1=P1,
        P2=P2,
        disp12MaxDiff=SGBM_DISP12MAX,
        uniquenessRatio=SGBM_UNIQUENESS,
        speckleWindowSize=SGBM_SPECKLE_WIN,
        speckleRange=SGBM_SPECKLE_RANGE,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # WLS 滤波器（提升视差图边缘质量）
    try:
        matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        use_wls = True
        print("[✓] WLS 视差滤波器已启用（需要 opencv-contrib-python）")
    except AttributeError:
        matcher_right = None
        wls_filter = None
        use_wls = False
        print("[!] WLS 滤波器不可用（仅 opencv-python），视差图质量略低")

    return matcher_left, matcher_right, wls_filter, use_wls


def compute_disparity(gray_left, gray_right,
                      matcher_left, matcher_right, wls_filter, use_wls,
                      rect_left=None):
    """
    计算视差图。
    若 WLS 可用，使用左右一致性检查 + WLS 滤波（更精准）；
    否则仅用左视差图。
    返回浮点视差图（单位：像素）
    """
    disp_left = matcher_left.compute(gray_left, gray_right)

    if use_wls and matcher_right is not None:
        disp_right = matcher_right.compute(gray_right, gray_left)
        filtered = wls_filter.filter(
            disp_left, rect_left, disparity_map_right=disp_right
        )
        disparity = filtered.astype(np.float32) / 16.0
    else:
        disparity = disp_left.astype(np.float32) / 16.0

    return disparity


def disparity_to_3d(Q, disparity):
    """
    使用 Q 矩阵将视差图重投影为 3D 点云。
    返回形状为 (H, W, 3) 的 XYZ 坐标图，单位：mm（与标定时一致）。
    """
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    return points_3d


def get_3d_coord_at_roi(points_3d, x1, y1, x2, y2, method="median"):
    """
    提取检测框 ROI 区域内的 3D 坐标。

    method:
      "center"  - 取检测框中心点的深度
      "median"  - 取有效点的中位数深度（最鲁棒）
      "nearest" - 取最近的有效点（适合抓取最前端）

    返回 (X, Y, Z) mm，或 None（若无有效点）
    """
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = points_3d.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    if method == "center":
        # 以中心点为准，若无效则在3×3邻域内搜索
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                py, px = cy + dy, cx + dx
                if 0 <= py < h and 0 <= px < w:
                    pt = points_3d[py, px]
                    Z = pt[2]
                    if MIN_DEPTH_MM < Z < MAX_DEPTH_MM:
                        return float(pt[0]), float(pt[1]), float(pt[2])
        return None

    # 取ROI区域内的有效点
    roi = points_3d[y1c:y2c, x1c:x2c]
    mask = (roi[:, :, 2] > MIN_DEPTH_MM) & (roi[:, :, 2] < MAX_DEPTH_MM)

    if not np.any(mask):
        return None

    valid_points = roi[mask]  # shape (N, 3)

    if method == "nearest":
        idx = np.argmin(valid_points[:, 2])
        pt = valid_points[idx]
    else:  # median
        pt = np.median(valid_points, axis=0)

    return float(pt[0]), float(pt[1]), float(pt[2])


def load_yolo_model(model_path):
    """加载 YOLO 模型（自动选择最佳可用模型）"""
    from ultralytics import YOLO

    # 优先使用自训练模型
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"[✓] 使用自训练模型: {TRAINED_MODEL_PATH}")
        return YOLO(TRAINED_MODEL_PATH)

    if os.path.exists(model_path):
        print(f"[✓] 使用通用模型: {model_path}")
        return YOLO(model_path)

    # 回退到基础模型
    fallback = "yolo11n.pt"
    print(f"[!] 未找到指定模型，回退使用: {fallback}")
    return YOLO(fallback)


def draw_3d_overlay(frame, detections, color=(0, 255, 0)):
    """
    在图像上绘制检测框和3D坐标标注。
    detections: list of dict，每个包含 bbox, label, conf, xyz
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf  = det["conf"]
        xyz   = det.get("xyz")

        # 画检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 画中心点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # 标注文字
        header = f"{label} {conf:.2f}"
        cv2.putText(frame, header, (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if xyz is not None:
            X, Y, Z = xyz
            coord_text = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"
            cv2.putText(frame, coord_text, (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

            # 深度用颜色渐变条表示（蓝=近, 红=远）
            depth_ratio = np.clip((Z - MIN_DEPTH_MM) / (MAX_DEPTH_MM - MIN_DEPTH_MM), 0, 1)
            bar_color = (
                int(255 * depth_ratio),        # B
                0,                              # G
                int(255 * (1 - depth_ratio)),  # R
            )
            bar_x = x2 + 5
            bar_h = int((y2 - y1) * (1 - depth_ratio))
            cv2.rectangle(frame,
                          (bar_x, y1 + (y2 - y1) - bar_h),
                          (bar_x + 8, y2),
                          bar_color, -1)
            cv2.rectangle(frame, (bar_x, y1), (bar_x + 8, y2), (200, 200, 200), 1)
        else:
            cv2.putText(frame, "深度无效", (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 128, 255), 2)

    return frame


def draw_disparity_preview(disparity):
    """将视差图转为可视化彩色图"""
    disp_vis = disparity.copy()
    disp_vis[disp_vis < 0] = 0
    disp_vis = (disp_vis / SGBM_NUM_DISPARITIES * 255).astype(np.uint8)
    return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


def run_stereo_3d_localize(args):
    """主循环：双目视觉 + YOLO 实时三维定位"""

    print("\n" + "=" * 65)
    print("  🎯 双目立体视觉三维定位系统  v1.0")
    print("  目标：气瓶 (gas_cylinder) 实时 XYZ 坐标输出")
    print("=" * 65)

    # ── 打开摄像头 ──────────────────────────────────────────────────
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 减少延迟

    image_size = (FRAME_WIDTH, FRAME_HEIGHT)

    # ── 加载标定参数 ─────────────────────────────────────────────────
    try:
        params, maps = load_stereo_params(PARAMS_FILE, MAPS_FILE, image_size)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    Q       = params["Q"].astype(np.float32)
    map1_l  = maps["map1_l"]
    map2_l  = maps["map2_l"]
    map1_r  = maps["map1_r"]
    map2_r  = maps["map2_r"]

    # ── 构建 SGBM 视差匹配器 ─────────────────────────────────────────
    matcher_l, matcher_r, wls_filter, use_wls = build_sgbm_matcher()

    # ── 加载 YOLO 模型 ───────────────────────────────────────────────
    model = load_yolo_model(args.model)
    class_names = model.names
    print(f"[✓] 检测类别: {class_names}")

    # ── 坐标平滑缓冲 ─────────────────────────────────────────────────
    coord_history = collections.deque(maxlen=SMOOTH_WINDOW)

    # ── 性能统计 ─────────────────────────────────────────────────────
    fps_timer  = time.time()
    fps_frames = 0
    fps_val    = 0.0

    print("\n  操作说明:")
    print("  [d]     - 切换视差图显示")
    print("  [r]     - 切换校正图显示")
    print("  [s]     - 截图保存")
    print("  [+/-]   - 调整视差范围")
    print("  [ESC/q] - 退出\n")

    show_disparity  = False
    show_rectified  = False
    frame_count     = 0

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()

        if not ret_l or not ret_r:
            print("[警告] 摄像头读取失败，重试...")
            time.sleep(0.05)
            continue

        frame_count += 1

        # ── 立体校正（消除畸变，对齐极线）───────────────────────────
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # ── 计算视差图 ───────────────────────────────────────────────
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        disparity = compute_disparity(
            gray_l, gray_r,
            matcher_l, matcher_r, wls_filter, use_wls,
            rect_left=rect_l
        )

        # ── 3D 重投影 ────────────────────────────────────────────────
        points_3d = disparity_to_3d(Q, disparity)

        # ── YOLO 检测（仅在左图上检测）──────────────────────────────
        results = model(
            rect_l,
            conf=args.conf,
            verbose=False,
            device="cuda:0" if args.gpu else "cpu",
        )

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # 过滤类别（若 TARGET_CLASS_ID >= 0 则只检测该类）
                if TARGET_CLASS_ID >= 0 and cls_id != TARGET_CLASS_ID:
                    continue

                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # 从3D点云中提取目标坐标
                xyz = get_3d_coord_at_roi(
                    points_3d, x1, y1, x2, y2, method="median"
                )

                if xyz is not None:
                    # 坐标平滑
                    coord_history.append(xyz)
                    if len(coord_history) >= 2:
                        hist = np.array(list(coord_history))
                        xyz_smooth = tuple(np.median(hist, axis=0).tolist())
                    else:
                        xyz_smooth = xyz
                else:
                    xyz_smooth = None

                label = class_names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox":  (x1, y1, x2, y2),
                    "label": label,
                    "conf":  conf_val,
                    "xyz":   xyz_smooth,
                    "cls_id": cls_id,
                })

        # ── 控制台输出 ───────────────────────────────────────────────
        if detections:
            for det in detections:
                xyz = det["xyz"]
                if xyz:
                    X, Y, Z = xyz
                    print(
                        f"[{det['label']}] conf={det['conf']:.2f}  "
                        f"X={X:+7.1f}mm  Y={Y:+7.1f}mm  Z={Z:7.1f}mm"
                    )

        # ── 可视化 ───────────────────────────────────────────────────
        if not args.no_display:
            display = rect_l.copy()

            if show_rectified:
                # 叠加极线网格
                for y_line in range(0, display.shape[0], 40):
                    cv2.line(display,
                             (0, y_line), (display.shape[1], y_line),
                             (0, 200, 0), 1)

            # 绘制检测框和3D坐标
            display = draw_3d_overlay(display, detections)

            # FPS 计算
            fps_frames += 1
            if time.time() - fps_timer >= 1.0:
                fps_val    = fps_frames / (time.time() - fps_timer)
                fps_frames = 0
                fps_timer  = time.time()

            cv2.putText(display, f"FPS: {fps_val:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display,
                        f"目标数: {len(detections)}  "
                        f"[d]视差 [r]校正 [s]截图 [ESC]退出",
                        (10, FRAME_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if show_disparity:
                disp_color = draw_disparity_preview(disparity)
                # 缩放到与主图一样大再拼接
                disp_small = cv2.resize(disp_color, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
                h, w = disp_small.shape[:2]
                display[0:h, FRAME_WIDTH - w:FRAME_WIDTH] = disp_small
                cv2.putText(display, "视差图", (FRAME_WIDTH - w + 5, h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Stereo 3D Localization | 双目三维定位", display)

        # ── 键盘交互 ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):             # ESC / q → 退出
            break
        elif key == ord('d'):                  # d → 切换视差图
            show_disparity = not show_disparity
        elif key == ord('r'):                  # r → 切换校正极线网格
            show_rectified = not show_rectified
        elif key == ord('s'):                  # s → 保存截图
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, display if not args.no_display else rect_l)
            print(f"[截图] 已保存: {fname}")
        elif key == ord('+') or key == ord('='):   # + → 增大视差范围
            SGBM_NUM_DISPARITIES = min(256, SGBM_NUM_DISPARITIES + 16)
            matcher_l, matcher_r, wls_filter, use_wls = build_sgbm_matcher()
            print(f"[调整] numDisparities → {SGBM_NUM_DISPARITIES}")
        elif key == ord('-'):                  # - → 减小视差范围
            SGBM_NUM_DISPARITIES = max(16, SGBM_NUM_DISPARITIES - 16)
            matcher_l, matcher_r, wls_filter, use_wls = build_sgbm_matcher()
            print(f"[调整] numDisparities → {SGBM_NUM_DISPARITIES}")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print("\n[完成] 程序已退出。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="双目立体视觉 + YOLO 三维定位系统"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"YOLO 模型路径（默认: {DEFAULT_MODEL_PATH}）"
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESHOLD,
        help=f"检测置信度阈值（默认: {CONF_THRESHOLD}）"
    )
    parser.add_argument(
        "--left", type=int, default=LEFT_CAM_INDEX,
        help=f"左摄像头设备索引（默认: {LEFT_CAM_INDEX}）"
    )
    parser.add_argument(
        "--right", type=int, default=RIGHT_CAM_INDEX,
        help=f"右摄像头设备索引（默认: {RIGHT_CAM_INDEX}）"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="无图形界面模式（纯命令行输出坐标，适合无显示器环境）"
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="使用 GPU 加速 YOLO 推理"
    )
    parser.add_argument(
        "--method", type=str, default="median",
        choices=["center", "median", "nearest"],
        help="ROI 内深度提取方法（默认: median）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 允许命令行覆盖摄像头索引
    LEFT_CAM_INDEX  = args.left
    RIGHT_CAM_INDEX = args.right

    try:
        run_stereo_3d_localize(args)
    except KeyboardInterrupt:
        print("\n[中断] 用户按下 Ctrl+C，程序退出。")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[异常] {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
