"""
环境检查脚本 | Environment Check Script
========================================
运行此脚本以验证所有依赖库已正确安装，并检查摄像头、GPU 等硬件是否可用。

使用方法：
  python check_env.py
  python check_env.py --cam       # 同时测试摄像头
  python check_env.py --full      # 完整检查（含模型加载）
"""

import sys
import os
import argparse
import time

# 脚本所在目录（localization/）和项目根目录（biyesheji/）
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

def _proj(relpath: str) -> str:
    """返回相对于项目根目录的绝对路径"""
    return os.path.join(_PROJECT_DIR, relpath)

def _local(relpath: str) -> str:
    """返回相对于本脚本所在目录的绝对路径"""
    return os.path.join(_SCRIPT_DIR, relpath)

# ── ANSI 颜色（Windows 需要先启用）─────────────────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}[✓]{RESET}"
WARN = f"{YELLOW}[!]{RESET}"
FAIL = f"{RED}[✗]{RESET}"
INFO = f"{CYAN}[i]{RESET}"


# ─────────────────────────────────────────────────────────────────────
# 检查函数
# ─────────────────────────────────────────────────────────────────────

def check_python():
    """检查 Python 版本"""
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 9:
        print(f"  {OK} Python {ver_str}")
        return True
    else:
        print(f"  {FAIL} Python {ver_str}  （需要 >= 3.9）")
        return False


def check_package(name: str, import_name: str = None,
                  version_attr: str = "__version__",
                  min_version: str = None) -> bool:
    """
    尝试导入一个包，并可选地检查版本。

    Args:
        name        : 包名（用于 pip install）
        import_name : 实际 import 时的名称（默认与 name 相同）
        version_attr: 版本属性名
        min_version : 最低版本字符串（如 "4.8.0"）

    Returns:
        True = 安装且版本满足；False = 未安装或版本不足
    """
    import_name = import_name or name

    try:
        mod = __import__(import_name)
        ver = getattr(mod, version_attr, "unknown")

        # 版本比较
        if min_version and ver != "unknown":
            try:
                from packaging.version import Version
                if Version(str(ver)) < Version(min_version):
                    print(f"  {WARN} {name:<30} v{ver}  （建议 >= {min_version}）")
                    return True   # 安装了但版本低，仍算通过（警告）
            except ImportError:
                pass  # packaging 未安装时跳过版本检查

        print(f"  {OK} {name:<30} v{ver}")
        return True

    except ImportError as e:
        print(f"  {FAIL} {name:<30} 未安装  →  pip install {name}")
        return False
    except Exception as e:
        print(f"  {WARN} {name:<30} 导入异常: {e}")
        return False


def check_opencv():
    """专门检查 OpenCV，区分基础版和 contrib 版"""
    try:
        import cv2
        ver = cv2.__version__
        # 检查是否有 ximgproc（contrib 模块，WLS 滤波需要）
        has_wls = hasattr(cv2, "ximgproc")
        if has_wls:
            print(f"  {OK} opencv-contrib-python        v{ver}  (含 WLS 滤波器)")
        else:
            print(f"  {WARN} opencv-python               v{ver}  "
                  f"（无 ximgproc，视差质量略低；建议安装 opencv-contrib-python）")
        return True
    except ImportError:
        print(f"  {FAIL} opencv-python/contrib          未安装  "
              f"→  pip install opencv-contrib-python")
        return False


def check_torch():
    """检查 PyTorch 及 CUDA 可用性"""
    try:
        import torch
        ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver   = torch.version.cuda if cuda_avail else "N/A"
        device_cnt = torch.cuda.device_count() if cuda_avail else 0

        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            print(f"  {OK} torch                          v{ver}")
            print(f"  {OK} CUDA                           v{cuda_ver}  "
                  f"({device_cnt} GPU: {device_name})")
        else:
            print(f"  {OK} torch                          v{ver}")
            print(f"  {WARN} CUDA                          不可用  "
                  f"（将使用 CPU，速度较慢）")
        return True
    except ImportError:
        print(f"  {FAIL} torch                          未安装  "
              f"→  pip install torch torchvision")
        return False


def check_ultralytics():
    """检查 YOLO 模型文件"""
    try:
        import ultralytics
        ver = ultralytics.__version__
        print(f"  {OK} ultralytics (YOLO)             v{ver}")

        # 检查项目中已有的模型文件（路径相对于项目根目录）
        model_files = [
            _proj("yolo11n.pt"),
            _proj("yolo11x.pt"),
            _proj("yolo11x-seg.pt"),
            _proj("yolo11n-seg.pt"),
            _proj("gas_dataset/weights/best.pt"),  # 训练后的模型
        ]
        found = []
        for mf in model_files:
            if os.path.exists(mf):
                size_mb = os.path.getsize(mf) / 1024 / 1024
                rel = os.path.relpath(mf, _SCRIPT_DIR)
                found.append(f"{rel} ({size_mb:.1f}MB)")

        if found:
            for f in found:
                print(f"    {INFO} 模型文件: {f}")
        else:
            print(f"    {WARN} 未找到本地模型文件（运行时将自动下载）")

        return True
    except ImportError:
        print(f"  {FAIL} ultralytics                    未安装  "
              f"→  pip install ultralytics")
        return False


def check_transformers():
    """检查 Depth Anything V2 所需的 transformers 库"""
    try:
        import transformers
        ver = transformers.__version__
        print(f"  {OK} transformers                   v{ver}")

        # 检查 timm（Depth Anything backbone）
        try:
            import timm
            print(f"  {OK} timm                           v{timm.__version__}")
        except ImportError:
            print(f"  {WARN} timm                          未安装（Depth Anything V2 需要）"
                  f"  →  pip install timm")

        return True
    except ImportError:
        print(f"  {WARN} transformers                   未安装（方案二 Depth Anything V2 需要）"
              f"  →  pip install transformers")
        return False


def check_stereo_calibration_files():
    """检查双目标定文件是否存在"""
    params_file = _local("stereo_params.npz")
    maps_file   = _local("stereo_maps.npz")
    calib_dir   = _local("calib_images")

    print()
    print(f"  {BOLD}标定文件状态:{RESET}")

    _pf = os.path.basename(params_file)
    _mf = os.path.basename(maps_file)
    _cd = os.path.basename(calib_dir)

    if os.path.exists(params_file):
        size_kb = os.path.getsize(params_file) / 1024
        print(f"    {OK} {_pf:<25} ({size_kb:.1f} KB)")
    else:
        print(f"    {WARN} {_pf:<25} 不存在  "
              f"→  先运行 stereo_calibration.py")

    if os.path.exists(maps_file):
        size_kb = os.path.getsize(maps_file) / 1024
        print(f"    {OK} {_mf:<25} ({size_kb:.1f} KB)")
    else:
        print(f"    {INFO} {_mf:<25} 不存在（标定后自动生成）")

    if os.path.isdir(calib_dir):
        left_dir  = os.path.join(calib_dir, "left")
        right_dir = os.path.join(calib_dir, "right")
        n_left  = len([f for f in os.listdir(left_dir)
                       if f.endswith(".jpg")]) if os.path.isdir(left_dir) else 0
        n_right = len([f for f in os.listdir(right_dir)
                       if f.endswith(".jpg")]) if os.path.isdir(right_dir) else 0
        print(f"    {INFO} {_cd:<25} 左:{n_left}张  右:{n_right}张")
    else:
        print(f"    {INFO} {_cd:<25} 不存在（采集时自动创建）")

    hand_eye = _local("hand_eye_params.npz")
    _he = os.path.basename(hand_eye)
    if os.path.exists(hand_eye):
        print(f"    {OK} {_he:<25} （Hand-Eye 标定完成）")
    else:
        print(f"    {WARN} {_he:<25} 不存在  "
              f"→  需完成 Hand-Eye 标定才能输出机械臂坐标")


def check_cameras(test_cam: bool = False):
    """检查摄像头可用性"""
    print()
    print(f"  {BOLD}摄像头检测:{RESET}")

    try:
        import cv2
    except ImportError:
        print(f"    {FAIL} 无法检测摄像头（OpenCV 未安装）")
        return

    found_cams = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found_cams.append((idx, w, h))
            cap.release()

    if len(found_cams) == 0:
        print(f"    {FAIL} 未检测到任何摄像头")
    elif len(found_cams) == 1:
        idx, w, h = found_cams[0]
        print(f"    {WARN} 仅检测到 1 个摄像头 (索引={idx}, {w}×{h})")
        print(f"         双目定位需要 2 个摄像头，请接入第二个摄像头。")
    else:
        for idx, w, h in found_cams:
            print(f"    {OK} 摄像头 #{idx}  分辨率: {w}×{h}")
        if len(found_cams) >= 2:
            print(f"    {OK} 双目摄像头配置满足要求（找到 {len(found_cams)} 个）")

    # 实时预览测试
    if test_cam and found_cams:
        print(f"\n    {INFO} 开启摄像头预览（按 ESC 关闭）...")
        idx = found_cams[0][0]
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        t_start = time.time()
        frames  = 0
        while time.time() - t_start < 5.0:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            fps = frames / (time.time() - t_start)
            cv2.putText(frame, f"Camera #{idx}  FPS: {fps:.1f}  (ESC=退出)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(f"摄像头 #{idx} 测试", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"    {OK} 摄像头 #{idx} 预览正常  (平均 FPS: {fps:.1f})")


def check_full_pipeline(args):
    """
    完整流水线测试：加载 YOLO 模型并对一张测试图片进行推理。
    """
    print()
    print(f"  {BOLD}完整推理测试:{RESET}")

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO

        # 选择测试模型（路径相对于项目根目录）
        model_path = None
        for candidate in [_proj("yolo11n.pt"), _proj("yolo11x.pt"),
                          _proj("yolo11x-seg.pt")]:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if model_path is None:
            print(f"    {WARN} 未找到本地模型文件，跳过推理测试。")
            print(f"         运行一次 stereo_3d_localize.py 会自动下载模型。")
            return

        print(f"    {INFO} 加载模型: {model_path}")
        model = YOLO(model_path)

        # 创建一张随机测试图片
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        results = model(test_img, verbose=False, conf=0.5)
        elapsed_ms = (time.time() - t0) * 1000

        print(f"    {OK} YOLO 推理成功  耗时: {elapsed_ms:.1f}ms")
        print(f"    {INFO} 检测到 {sum(len(r.boxes) for r in results)} 个目标（随机图像，预期为0）")

    except Exception as e:
        print(f"    {FAIL} 推理测试失败: {e}")


def check_utils():
    """检查本模块的工具代码是否可以正常导入"""
    print()
    print(f"  {BOLD}本地模块检查:{RESET}")

    # 把 localization 目录加入 sys.path，使 utils 可以正确导入
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)

    modules_to_check = [
        ("utils.coord_transform", "CoordTransformer"),
    ]

    for mod_path, cls_name in modules_to_check:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            print(f"    {OK} {mod_path}.{cls_name}")
        except ImportError as e:
            print(f"    {FAIL} {mod_path}  →  {e}")
        except Exception as e:
            print(f"    {WARN} {mod_path}  →  {e}")


def print_quick_start():
    """打印快速上手指南"""
    print()
    print(f"  {BOLD}{'='*55}{RESET}")
    print(f"  {BOLD}  快速上手指南{RESET}")
    print(f"  {BOLD}{'='*55}{RESET}")
    print(f"""
  {BOLD}【方案一：双目立体视觉（推荐）】{RESET}
  Step 1  完成双目标定
          python stereo_calibration.py
          （需要两个摄像头 + 打印棋盘格）

  Step 2  实时三维定位
          python stereo_3d_localize.py
          （按 [d] 查看视差图，[s] 截图）

  {BOLD}【方案二：单目深度估计（无需双目）】{RESET}
  直接运行（首次会下载约100MB模型）：
          python depth_anything_localize.py
          python depth_anything_localize.py --metric  # metric深度（直接输出米）
          python depth_anything_localize.py --calibrate 800  # 比例标定

  {BOLD}【方案三：多摄像头三角测量】{RESET}
  需先完成方案一的标定，然后：
          python triangulation_3d.py --show-epilines

  {BOLD}坐标系转换（集成机械臂）：{RESET}
          from utils.coord_transform import CoordTransformer
          tf = CoordTransformer("hand_eye_params.npz")
          x_r, y_r, z_r = tf.camera_to_robot(x_c, y_c, z_c)
""")


# ─────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3D 定位系统 - 环境依赖检查工具"
    )
    parser.add_argument(
        "--cam", action="store_true",
        help="测试摄像头（会短暂打开预览窗口）"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="完整检查（含 YOLO 模型加载推理测试）"
    )
    parser.add_argument(
        "--no-guide", action="store_true",
        help="不显示快速上手指南"
    )
    args = parser.parse_args()

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  🔍 3D 定位系统 - 环境依赖检查{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    all_ok = True

    # ── Python ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[1] Python 版本{RESET}")
    all_ok &= check_python()

    # ── 核心依赖 ────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] 核心依赖库{RESET}")
    all_ok &= check_opencv()
    all_ok &= check_torch()
    all_ok &= check_ultralytics()
    all_ok &= check_package("numpy",      "numpy",      min_version="1.24.0")
    all_ok &= check_package("Pillow",     "PIL",        version_attr="__version__")
    all_ok &= check_package("scipy",      "scipy",      min_version="1.11.0")
    all_ok &= check_package("matplotlib", "matplotlib", min_version="3.7.0")
    all_ok &= check_package("tqdm",       "tqdm")

    # ── 可选依赖 ────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] 可选依赖（方案二：Depth Anything V2）{RESET}")
    check_transformers()

    print(f"\n{BOLD}[4] 可选依赖（3D 点云可视化）{RESET}")
    try:
        import open3d
        print(f"  {OK} open3d                         v{open3d.__version__}")
    except ImportError:
        print(f"  {WARN} open3d                         未安装（可选）"
              f"  →  pip install open3d")

    # ── 标定文件 ────────────────────────────────────────────────────
    print(f"\n{BOLD}[5] 标定文件 & 数据{RESET}")
    check_stereo_calibration_files()

    # ── 摄像头 ──────────────────────────────────────────────────────
    print(f"\n{BOLD}[6] 硬件设备{RESET}")
    check_cameras(test_cam=args.cam)

    # ── 本地模块 ────────────────────────────────────────────────────
    print(f"\n{BOLD}[7] 本地模块{RESET}")
    check_utils()

    # ── 完整推理测试 ─────────────────────────────────────────────────
    if args.full:
        print(f"\n{BOLD}[8] 完整推理测试{RESET}")
        check_full_pipeline(args)

    # ── 总结 ────────────────────────────────────────────────────────
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    if all_ok:
        print(f"{BOLD}{GREEN}  ✅ 核心依赖检查通过，可以开始运行！{RESET}")
    else:
        print(f"{BOLD}{RED}  ⚠️  部分依赖缺失，请按上方提示安装后重试。{RESET}")
        print(f"\n  一键安装命令：")
        print(f"  {CYAN}pip install opencv-contrib-python ultralytics "
              f"transformers timm torch torchvision numpy scipy matplotlib tqdm{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # ── 快速上手指南 ─────────────────────────────────────────────────
    if not args.no_guide:
        print_quick_start()


if __name__ == "__main__":
    main()
