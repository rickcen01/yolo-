"""
双目相机标定脚本 | Stereo Camera Calibration
=============================================
使用棋盘格对两个相机进行标定，得到：
  - 左/右相机内参矩阵 (intrinsics)
  - 左/右相机畸变系数 (distortion)
  - 两相机间旋转矩阵 R 和平移向量 T
  - 用于视差→深度转换的重投影矩阵 Q

使用方法：
  1. 准备一张 9×6 的棋盘格（或修改下方 CHESSBOARD_SIZE）
  2. 将两个摄像头连接好并固定（不能移动！）
  3. 运行脚本：python stereo_calibration.py
  4. 按 [空格] 采集，按 [ESC] 结束采集并开始计算
  5. 结果保存到 stereo_params.npz

依赖：opencv-contrib-python, numpy
"""

import cv2
import numpy as np
import os
import time
import glob

# =====================================================================
# ⚙️  配置区域 - 根据实际情况修改
# =====================================================================

# 棋盘格内角点数量 (列数-1, 行数-1)
# 例如：9×6 的棋盘格，内角点为 (8, 5)
CHESSBOARD_SIZE = (8, 5)

# 棋盘格每个格子的实际边长（单位：mm）
SQUARE_SIZE_MM = 25.0

# 左摄像头设备索引（通常 0 是内置摄像头，1、2 是外接）
LEFT_CAM_INDEX = 0

# 右摄像头设备索引
RIGHT_CAM_INDEX = 1

# 采集图像保存目录（用于调试）
CALIB_IMAGES_DIR = "calib_images"

# 标定参数保存文件名
OUTPUT_PARAMS_FILE = "stereo_params.npz"

# 最少需要采集的有效帧数
MIN_VALID_FRAMES = 20

# 采集间隔（秒）：按空格后需等待的时间（防止连拍太多重复帧）
CAPTURE_COOLDOWN = 0.5

# 图像分辨率（None 表示使用相机默认分辨率）
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =====================================================================


def create_object_points(chessboard_size, square_size):
    """
    生成棋盘格的3D世界坐标点（Z=0 平面上）
    返回形如 [(0,0,0), (25,0,0), (50,0,0), ...] 的坐标数组
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0:chessboard_size[0], 0:chessboard_size[1]
    ].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_chessboard_corners(img_gray, chessboard_size):
    """
    检测棋盘格角点，并进行亚像素级精化
    返回 (found, corners)
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(img_gray, chessboard_size, flags)

    if found:
        # 亚像素精化（提升标定精度）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(
            img_gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
        )
    return found, corners


def draw_corners_preview(img, chessboard_size, found, corners):
    """在图像上绘制检测到的棋盘格角点（用于预览）"""
    preview = img.copy()
    cv2.drawChessboardCorners(preview, chessboard_size, corners, found)
    return preview


def collect_calibration_images(left_cam_idx, right_cam_idx, chessboard_size):
    """
    交互式采集双目标定图像
    按 [空格] 保存当前帧，按 [ESC] 结束采集
    返回：(obj_points_list, img_points_left, img_points_right, image_size)
    """
    os.makedirs(os.path.join(CALIB_IMAGES_DIR, "left"), exist_ok=True)
    os.makedirs(os.path.join(CALIB_IMAGES_DIR, "right"), exist_ok=True)

    cap_left = cv2.VideoCapture(left_cam_idx)
    cap_right = cv2.VideoCapture(right_cam_idx)

    if not cap_left.isOpened():
        raise RuntimeError(f"无法打开左摄像头 (索引={left_cam_idx})，请检查设备连接。")
    if not cap_right.isOpened():
        raise RuntimeError(f"无法打开右摄像头 (索引={right_cam_idx})，请检查设备连接。")

    # 设置分辨率
    for cap in [cap_left, cap_right]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    objp = create_object_points(chessboard_size, SQUARE_SIZE_MM)

    obj_points_list = []
    img_points_left = []
    img_points_right = []
    image_size = None

    valid_count = 0
    last_capture_time = 0
    frame_idx = 0

    print("\n" + "=" * 60)
    print("📷 双目标定图像采集模式")
    print("=" * 60)
    print(f"  棋盘格规格：{chessboard_size[0]+1} × {chessboard_size[1]+1} 格")
    print(f"  格子边长：{SQUARE_SIZE_MM} mm")
    print(f"  目标采集帧数：>= {MIN_VALID_FRAMES} 帧")
    print()
    print("  操作说明：")
    print("  [空格]  - 当两侧都检测到棋盘格时，保存当前帧")
    print("  [ESC]   - 结束采集，开始计算标定参数")
    print("  [q]     - 退出程序")
    print("=" * 60 + "\n")

    while True:
        ret_l, frame_left = cap_left.read()
        ret_r, frame_right = cap_right.read()

        if not ret_l or not ret_r:
            print("[警告] 无法读取摄像头帧，跳过...")
            time.sleep(0.1)
            continue

        if image_size is None:
            h, w = frame_left.shape[:2]
            image_size = (w, h)

        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_chessboard_corners(gray_left, chessboard_size)
        found_r, corners_r = find_chessboard_corners(gray_right, chessboard_size)

        # 绘制检测结果
        display_left = frame_left.copy()
        display_right = frame_right.copy()

        status_color_l = (0, 255, 0) if found_l else (0, 0, 255)
        status_color_r = (0, 255, 0) if found_r else (0, 0, 255)
        status_text_l = "✓ 检测到" if found_l else "✗ 未检测到"
        status_text_r = "✓ 检测到" if found_r else "✗ 未检测到"

        if found_l:
            cv2.drawChessboardCorners(display_left, chessboard_size, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(display_right, chessboard_size, corners_r, found_r)

        # 添加状态文字
        cv2.putText(display_left, f"LEFT  {status_text_l}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color_l, 2)
        cv2.putText(display_right, f"RIGHT {status_text_r}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color_r, 2)
        cv2.putText(display_left, f"已采集: {valid_count}/{MIN_VALID_FRAMES}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_right, "空格=保存  ESC=完成", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 拼接左右图像显示
        combined = np.hstack([display_left, display_right])
        cv2.imshow("Stereo Calibration - LEFT | RIGHT", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print(f"\n[采集结束] 共采集有效帧: {valid_count} 帧")
            break
        elif key == ord('q'):
            print("[退出] 用户主动退出。")
            cap_left.release()
            cap_right.release()
            cv2.destroyAllWindows()
            return None, None, None, None
        elif key == ord(' '):  # 空格
            now = time.time()
            if now - last_capture_time < CAPTURE_COOLDOWN:
                continue

            if found_l and found_r:
                obj_points_list.append(objp)
                img_points_left.append(corners_l)
                img_points_right.append(corners_r)
                valid_count += 1
                last_capture_time = now

                # 保存图像到磁盘
                left_path = os.path.join(CALIB_IMAGES_DIR, "left", f"frame_{frame_idx:04d}.jpg")
                right_path = os.path.join(CALIB_IMAGES_DIR, "right", f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(left_path, frame_left)
                cv2.imwrite(right_path, frame_right)
                frame_idx += 1

                print(f"  [✓ 帧 {valid_count:3d}] 已保存: {left_path}")

                # 短暂闪烁反馈
                flash = np.ones_like(combined) * 255
                cv2.imshow("Stereo Calibration - LEFT | RIGHT", flash)
                cv2.waitKey(150)
            else:
                missing = []
                if not found_l:
                    missing.append("左")
                if not found_r:
                    missing.append("右")
                print(f"  [✗ 跳过] {'、'.join(missing)}摄像头未检测到棋盘格，请调整角度。")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    return obj_points_list, img_points_left, img_points_right, image_size


def load_from_saved_images(chessboard_size):
    """
    从已保存的图像目录重新加载标定点（无需重新采集）
    """
    left_dir = os.path.join(CALIB_IMAGES_DIR, "left")
    right_dir = os.path.join(CALIB_IMAGES_DIR, "right")

    left_images = sorted(glob.glob(os.path.join(left_dir, "*.jpg")))
    right_images = sorted(glob.glob(os.path.join(right_dir, "*.jpg")))

    if len(left_images) == 0 or len(right_images) == 0:
        return None, None, None, None

    if len(left_images) != len(right_images):
        print(f"[警告] 左图数量 ({len(left_images)}) ≠ 右图数量 ({len(right_images)})")

    objp = create_object_points(chessboard_size, SQUARE_SIZE_MM)
    obj_points_list = []
    img_points_left = []
    img_points_right = []
    image_size = None

    print(f"\n[重新加载] 从 {CALIB_IMAGES_DIR} 加载 {len(left_images)} 对图像...")

    for lp, rp in zip(left_images, right_images):
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            continue

        if image_size is None:
            h, w = img_l.shape[:2]
            image_size = (w, h)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_chessboard_corners(gray_l, chessboard_size)
        found_r, corners_r = find_chessboard_corners(gray_r, chessboard_size)

        if found_l and found_r:
            obj_points_list.append(objp)
            img_points_left.append(corners_l)
            img_points_right.append(corners_r)
            print(f"  [✓] {os.path.basename(lp)}")
        else:
            print(f"  [✗] {os.path.basename(lp)} - 跳过（检测失败）")

    print(f"[加载完成] 有效帧: {len(obj_points_list)} / {len(left_images)}")
    return obj_points_list, img_points_left, img_points_right, image_size


def run_stereo_calibration(obj_points, img_pts_l, img_pts_r, image_size):
    """
    执行双目标定（包含单目标定 + 双目联合标定）
    返回标定结果字典
    """
    print("\n" + "=" * 60)
    print("🔧 开始计算标定参数...")
    print("=" * 60)

    calib_flags = (
        cv2.CALIB_RATIONAL_MODEL  # 使用高精度畸变模型（8参数）
    )

    # Step 1: 单独标定左相机
    print("\n[1/3] 标定左相机...")
    rms_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_pts_l, image_size, None, None
    )
    print(f"      左相机重投影误差 (RMS): {rms_l:.4f} 像素")

    # Step 2: 单独标定右相机
    print("[2/3] 标定右相机...")
    rms_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_pts_r, image_size, None, None
    )
    print(f"      右相机重投影误差 (RMS): {rms_r:.4f} 像素")

    # Step 3: 双目联合标定
    print("[3/3] 双目联合标定...")
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC  # 固定单目标定结果，只优化 R, T
    )
    rms_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_pts_l,
        img_pts_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=stereo_flags,
    )
    print(f"      双目联合重投影误差 (RMS): {rms_stereo:.4f} 像素")

    if rms_stereo > 1.0:
        print(f"\n  ⚠️  警告：RMS > 1.0 像素，标定质量较差！")
        print(f"      建议：增加采集帧数，确保棋盘格覆盖画面边缘和不同角度。")
    elif rms_stereo < 0.5:
        print(f"\n  ✅  标定质量优秀（RMS < 0.5 像素）")
    else:
        print(f"\n  ✅  标定质量良好")

    # Step 4: 计算立体校正参数（消除畸变，对齐极线）
    print("\n[+] 计算立体校正矩阵 (stereoRectify)...")
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        mtx_l, dist_l,
        mtx_r, dist_r,
        image_size,
        R, T,
        alpha=0,           # alpha=0: 裁剪所有无效像素；alpha=1: 保留全部
        newImageSize=(0, 0)
    )

    # 基线距离（mm）
    baseline_mm = abs(T[0][0])
    focal_px = P1[0, 0]  # 校正后焦距（像素）

    print(f"\n  📏 基线距离 (Baseline): {baseline_mm:.2f} mm")
    print(f"  🔭 焦距 (Focal Length): {focal_px:.2f} 像素")
    print(f"  📐 图像尺寸: {image_size[0]} × {image_size[1]}")

    return {
        "mtx_left": mtx_l,
        "dist_left": dist_l,
        "mtx_right": mtx_r,
        "dist_right": dist_r,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "image_size": image_size,
        "rms_stereo": rms_stereo,
        "baseline_mm": baseline_mm,
        "focal_px": focal_px,
    }


def save_params(params, output_file):
    """保存标定参数到 .npz 文件"""
    np.savez(
        output_file,
        mtx_left=params["mtx_left"],
        dist_left=params["dist_left"],
        mtx_right=params["mtx_right"],
        dist_right=params["dist_right"],
        R=params["R"],
        T=params["T"],
        E=params["E"],
        F=params["F"],
        R1=params["R1"],
        R2=params["R2"],
        P1=params["P1"],
        P2=params["P2"],
        Q=params["Q"],
        roi_left=params["roi_left"],
        roi_right=params["roi_right"],
        image_size=params["image_size"],
        baseline_mm=params["baseline_mm"],
        focal_px=params["focal_px"],
    )
    print(f"\n[💾 保存完成] 标定参数已保存到: {output_file}")


def verify_calibration(params, image_size):
    """
    验证标定结果：生成校正映射并展示极线对齐效果
    """
    mtx_l = params["mtx_left"]
    dist_l = params["dist_left"]
    mtx_r = params["mtx_right"]
    dist_r = params["dist_right"]
    R1, R2, P1, P2 = params["R1"], params["R2"], params["P1"], params["P2"]

    # 生成校正映射表
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
    )

    # 保存映射表（供后续脚本使用）
    np.savez(
        "stereo_maps.npz",
        map1_l=map1_l,
        map2_l=map2_l,
        map1_r=map1_r,
        map2_r=map2_r,
    )
    print("[💾 保存完成] 校正映射已保存到: stereo_maps.npz")

    print("\n[验证] 打开摄像头预览极线对齐效果（按 ESC 退出）...")
    cap_l = cv2.VideoCapture(LEFT_CAM_INDEX)
    cap_r = cv2.VideoCapture(RIGHT_CAM_INDEX)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("[跳过验证] 摄像头不可用。")
        return

    for cap in [cap_l, cap_r]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # 应用校正
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # 拼接并画极线（水平线应对齐）
        combined = np.hstack([rect_l, rect_r])
        h = combined.shape[0]
        for y in range(0, h, 40):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

        cv2.putText(combined, "校正预览 - 绿线应与物体边缘对齐 | ESC退出",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Stereo Rectification Verification", combined)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def print_calibration_summary(params):
    """打印标定参数摘要"""
    print("\n" + "=" * 60)
    print("📊 标定参数摘要")
    print("=" * 60)

    mtx_l = params["mtx_left"]
    mtx_r = params["mtx_right"]
    T = params["T"]

    print(f"\n  左相机内参:")
    print(f"    fx={mtx_l[0,0]:.2f}  fy={mtx_l[1,1]:.2f}")
    print(f"    cx={mtx_l[0,2]:.2f}  cy={mtx_l[1,2]:.2f}")

    print(f"\n  右相机内参:")
    print(f"    fx={mtx_r[0,0]:.2f}  fy={mtx_r[1,1]:.2f}")
    print(f"    cx={mtx_r[0,2]:.2f}  cy={mtx_r[1,2]:.2f}")

    print(f"\n  基线 (Baseline): {params['baseline_mm']:.2f} mm")
    print(f"  平移向量 T: [{T[0][0]:.2f}, {T[1][0]:.2f}, {T[2][0]:.2f}] mm")
    print(f"\n  双目RMS误差: {params['rms_stereo']:.4f} 像素")
    print(f"\n  Q矩阵（视差→3D）:\n{params['Q']}")
    print("\n" + "=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  🎯 双目相机标定工具 v1.0")
    print("  适用于气瓶三维定位系统")
    print("=" * 60)

    # 检查是否已有保存的标定图像，提供选项
    left_dir = os.path.join(CALIB_IMAGES_DIR, "left")
    has_saved = os.path.isdir(left_dir) and len(glob.glob(os.path.join(left_dir, "*.jpg"))) > 0

    mode = "capture"
    if has_saved:
        saved_count = len(glob.glob(os.path.join(left_dir, "*.jpg")))
        print(f"\n  检测到已保存的标定图像 ({saved_count} 对)")
        print("  请选择模式:")
        print("  [1] 重新采集（覆盖已有图像）")
        print("  [2] 使用已保存图像重新计算")
        choice = input("  输入选项 (1/2): ").strip()
        if choice == "2":
            mode = "reload"

    if mode == "capture":
        print(f"\n  配置:")
        print(f"    左摄像头索引: {LEFT_CAM_INDEX}")
        print(f"    右摄像头索引: {RIGHT_CAM_INDEX}")
        print(f"    棋盘格: {CHESSBOARD_SIZE[0]+1}×{CHESSBOARD_SIZE[1]+1} 格，边长 {SQUARE_SIZE_MM}mm")
        input("\n  按 Enter 开始采集...")

        obj_points, img_pts_l, img_pts_r, image_size = collect_calibration_images(
            LEFT_CAM_INDEX, RIGHT_CAM_INDEX, CHESSBOARD_SIZE
        )
    else:
        obj_points, img_pts_l, img_pts_r, image_size = load_from_saved_images(CHESSBOARD_SIZE)

    if obj_points is None or len(obj_points) < MIN_VALID_FRAMES:
        n = len(obj_points) if obj_points else 0
        print(f"\n❌ 有效帧数不足（{n}/{MIN_VALID_FRAMES}），无法进行标定。")
        print("   请重新采集更多棋盘格图像。")
        return

    # 执行标定
    params = run_stereo_calibration(obj_points, img_pts_l, img_pts_r, image_size)

    # 打印摘要
    print_calibration_summary(params)

    # 保存参数
    save_params(params, OUTPUT_PARAMS_FILE)

    # 验证校正效果
    print("\n  是否进行校正效果验证（需要摄像头在线）？")
    verify = input("  输入 y/n: ").strip().lower()
    if verify == 'y':
        verify_calibration(params, tuple(params["image_size"]))

    print("\n✅ 标定流程完成！")
    print(f"   下一步：运行 stereo_3d_localize.py 进行实时三维定位")
    print()


if __name__ == "__main__":
    main()
