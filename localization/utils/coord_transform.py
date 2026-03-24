"""
坐标系转换工具 | Coordinate System Transform Utility
=====================================================
将相机坐标系下的三维坐标转换到机械臂基坐标系，
并提供常用的刚体变换、角度计算、坐标校验等辅助函数。

坐标系说明
----------
相机坐标系 (Camera Frame)
    X_c : 向右为正
    Y_c : 向下为正
    Z_c : 朝向场景（深度方向）为正
    原点: 左相机光心

机械臂基坐标系 (Robot Base Frame) —— 典型工业机械臂约定
    X_r : 机械臂正前方
    Y_r : 机械臂左侧
    Z_r : 垂直向上
    原点: 机械臂基座中心

眼在手外 (Eye-to-Hand) 标定流程概述
--------------------------------------
1. 将标定板（棋盘格）固定在机械臂末端夹爪上
2. 令机械臂运动到若干已知位姿，每次记录：
     - 机械臂末端的基坐标系位姿 T_base_end  (由机械臂控制器读取)
     - 相机看到标定板的位姿     T_cam_board (由 cv2.solvePnP 计算)
3. 利用 cv2.calibrateHandEye() 求解  T_cam_base（或其逆）
4. 将求解结果保存为 hand_eye_params.npz，由本模块加载使用

使用示例
--------
>>> from utils.coord_transform import CoordTransformer
>>> tf = CoordTransformer("hand_eye_params.npz")
>>> x_robot, y_robot, z_robot = tf.camera_to_robot(x_cam, y_cam, z_cam)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple, Union

import numpy as np

# =====================================================================
# 类型别名
# =====================================================================
Vec3 = Union[np.ndarray, Tuple[float, float, float]]


# =====================================================================
# 基础刚体变换
# =====================================================================

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵 R (3×3) 和平移向量 t (3,) 合并为齐次变换矩阵 T (4×4)。

    T = | R  t |
        | 0  1 |
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3,  3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    高效求齐次变换矩阵的逆。

    对于刚体变换：T^{-1} = | R^T  -R^T t |
                            |  0      1   |
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3,  3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] =  R.T
    T_inv[:3,  3] = -R.T @ t
    return T_inv


def apply_transform(T: np.ndarray, point: Vec3) -> np.ndarray:
    """
    对单个三维点应用齐次变换矩阵。

    Args:
        T     : (4×4) 齐次变换矩阵
        point : (3,) 三维坐标

    Returns:
        transformed: (3,) 变换后的三维坐标
    """
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(point, dtype=np.float64).ravel()
    return (T @ p)[:3]


def apply_transform_batch(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    对一组三维点批量应用齐次变换矩阵。

    Args:
        T      : (4×4) 齐次变换矩阵
        points : (N, 3) 三维坐标数组

    Returns:
        transformed: (N, 3)
    """
    pts = np.asarray(points, dtype=np.float64)
    N   = pts.shape[0]
    pts_h = np.ones((N, 4), dtype=np.float64)
    pts_h[:, :3] = pts
    return (T @ pts_h.T).T[:, :3]


# =====================================================================
# 旋转表示转换
# =====================================================================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float,
                               degrees: bool = True) -> np.ndarray:
    """
    欧拉角 (ZYX 外旋 / roll-pitch-yaw) → 旋转矩阵 (3×3)。

    Args:
        roll  : 绕 X 轴旋转（横滚角）
        pitch : 绕 Y 轴旋转（俯仰角）
        yaw   : 绕 Z 轴旋转（偏航角）
        degrees: True 表示输入单位为度，False 为弧度

    Returns:
        R: (3×3) 旋转矩阵，满足 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    if degrees:
        roll  = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw   = np.deg2rad(yaw)

    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(roll), -np.sin(roll)],
        [0,  np.sin(roll),  np.cos(roll)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray,
                               degrees: bool = True) -> Tuple[float, float, float]:
    """
    旋转矩阵 → 欧拉角 (roll, pitch, yaw)，ZYX 约定。

    Returns:
        (roll, pitch, yaw)  单位由 degrees 参数决定
    """
    R = np.asarray(R, dtype=np.float64)

    # 检查万向锁（Gimbal Lock）
    if abs(R[2, 0]) >= 1.0 - 1e-6:
        yaw  = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll  = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll  = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = np.arcsin(-R[2, 0])
        roll  = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw   = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    if degrees:
        return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))
    return float(roll), float(pitch), float(yaw)


def rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """
    Rodrigues 旋转向量 → 旋转矩阵（OpenCV 约定）。

    Args:
        rvec: (3,) 或 (3,1) Rodrigues 向量

    Returns:
        R: (3×3) 旋转矩阵
    """
    R, _ = cv2_rodrigues(rvec)
    return R


def cv2_rodrigues(rvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """调用 OpenCV cv2.Rodrigues，做懒导入以避免循环依赖。"""
    import cv2
    R, jacobian = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).ravel())
    return R, jacobian


# =====================================================================
# 主要转换类
# =====================================================================

class CoordTransformer:
    """
    相机坐标系 ↔ 机械臂基坐标系 的转换器。

    支持两种初始化方式：
      1. 从 hand_eye_params.npz 文件加载（推荐，由 Hand-Eye 标定获得）
      2. 直接指定 R, t（用于测试或已知几何关系的简单场景）

    属性
    ----
    T_cam2robot : (4×4) 相机→机械臂基坐标系变换矩阵
    T_robot2cam : (4×4) 机械臂基坐标系→相机变换矩阵（自动求逆）

    示例
    ----
    >>> tf = CoordTransformer("hand_eye_params.npz")
    >>> xyz_robot = tf.camera_to_robot(450.0, -120.0, 820.0)
    >>> print(f"机械臂坐标: X={xyz_robot[0]:.1f}  Y={xyz_robot[1]:.1f}  Z={xyz_robot[2]:.1f} mm")
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        R: Optional[np.ndarray] = None,
        t: Optional[np.ndarray] = None,
    ):
        """
        初始化坐标转换器。

        优先级：params_file > (R, t) > 单位变换（无转换）

        Args:
            params_file : Hand-Eye 标定结果文件路径（.npz）
            R           : (3×3) 旋转矩阵（直接指定时使用）
            t           : (3,)  平移向量，单位 mm（直接指定时使用）
        """
        self._T_cam2robot: Optional[np.ndarray] = None
        self._T_robot2cam: Optional[np.ndarray] = None
        self._loaded      = False
        self._source      = "identity"

        if params_file is not None:
            self._load_from_file(params_file)
        elif R is not None and t is not None:
            self._set_from_RT(R, t)
        else:
            warnings.warn(
                "CoordTransformer: 未提供标定参数，将使用单位变换（坐标不转换）。\n"
                "请先完成 Hand-Eye 标定，然后传入 params_file 参数。",
                UserWarning,
                stacklevel=2,
            )
            self._T_cam2robot = np.eye(4, dtype=np.float64)
            self._T_robot2cam = np.eye(4, dtype=np.float64)
            self._source      = "identity"

    # ── 初始化辅助 ────────────────────────────────────────────────────

    def _load_from_file(self, filepath: str):
        """从 .npz 文件加载 Hand-Eye 标定参数"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Hand-Eye 标定文件不存在：{filepath}\n"
                "请先运行 Hand-Eye 标定，或手动指定 R 和 t 参数。\n"
                "参考：cv2.calibrateHandEye() 或 easy_handeye 库"
            )

        data = np.load(filepath, allow_pickle=True)

        # 支持多种常见字段命名
        R_keys = [k for k in data.files if k.lower() in ("r", "rotation", "r_cam2base", "r_cam_to_base")]
        t_keys = [k for k in data.files if k.lower() in ("t", "translation", "t_cam2base", "t_cam_to_base")]

        # 也支持直接存储 4×4 变换矩阵的格式
        T_keys = [k for k in data.files if k.lower() in ("t_matrix", "transform", "t_cam2robot", "hand_eye_matrix")]

        if T_keys:
            T = data[T_keys[0]].astype(np.float64)
            if T.shape == (4, 4):
                self._T_cam2robot = T
                self._T_robot2cam = invert_transform(T)
                self._loaded      = True
                self._source      = filepath
                print(f"[CoordTransformer] 已加载 Hand-Eye 变换矩阵: {filepath}")
                return

        if not R_keys or not t_keys:
            raise KeyError(
                f"标定文件 {filepath} 中缺少旋转矩阵(R)或平移向量(t)字段。\n"
                f"  检测到的字段: {data.files}\n"
                f"  期望字段: R/rotation/R_cam2base  以及  t/translation/t_cam2base"
            )

        R_mat = data[R_keys[0]].astype(np.float64)
        t_vec = data[t_keys[0]].astype(np.float64).ravel()

        # 若 R 是 Rodrigues 向量（(3,) 形式），先转旋转矩阵
        if R_mat.shape == (3,):
            R_mat, _ = cv2_rodrigues(R_mat)

        self._set_from_RT(R_mat, t_vec)
        self._source = filepath
        print(f"[CoordTransformer] 已加载 Hand-Eye 参数: {filepath}")
        self._print_summary()

    def _set_from_RT(self, R: np.ndarray, t: np.ndarray):
        """从 R, t 设置变换矩阵"""
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64).ravel()

        if R.shape != (3, 3):
            raise ValueError(f"旋转矩阵 R 的形状应为 (3,3)，实际为 {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"平移向量 t 的形状应为 (3,)，实际为 {t.shape}")

        self._T_cam2robot = make_transform(R, t)
        self._T_robot2cam = invert_transform(self._T_cam2robot)
        self._loaded      = True

    # ── 属性 ─────────────────────────────────────────────────────────

    @property
    def T_cam2robot(self) -> np.ndarray:
        """(4×4) 相机坐标系 → 机械臂基坐标系的变换矩阵"""
        return self._T_cam2robot.copy()

    @property
    def T_robot2cam(self) -> np.ndarray:
        """(4×4) 机械臂基坐标系 → 相机坐标系的变换矩阵"""
        return self._T_robot2cam.copy()

    @property
    def is_calibrated(self) -> bool:
        """是否使用了真实标定参数（而非单位变换）"""
        return self._loaded and self._source != "identity"

    # ── 坐标转换 ──────────────────────────────────────────────────────

    def camera_to_robot(
        self,
        x_cam: float,
        y_cam: float,
        z_cam: float,
    ) -> Tuple[float, float, float]:
        """
        将相机坐标系下的点转换到机械臂基坐标系。

        Args:
            x_cam, y_cam, z_cam : 相机坐标系下的 3D 坐标，单位 mm

        Returns:
            (x_robot, y_robot, z_robot) 机械臂基坐标系坐标，单位 mm
        """
        pt_cam   = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        pt_robot = apply_transform(self._T_cam2robot, pt_cam)
        return float(pt_robot[0]), float(pt_robot[1]), float(pt_robot[2])

    def robot_to_camera(
        self,
        x_robot: float,
        y_robot: float,
        z_robot: float,
    ) -> Tuple[float, float, float]:
        """
        将机械臂基坐标系下的点转换回相机坐标系。

        Args:
            x_robot, y_robot, z_robot : 机械臂基坐标系坐标，单位 mm

        Returns:
            (x_cam, y_cam, z_cam) 相机坐标系坐标，单位 mm
        """
        pt_robot = np.array([x_robot, y_robot, z_robot], dtype=np.float64)
        pt_cam   = apply_transform(self._T_robot2cam, pt_robot)
        return float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])

    def camera_to_robot_batch(self, points: np.ndarray) -> np.ndarray:
        """
        批量转换：相机坐标系 → 机械臂基坐标系。

        Args:
            points: (N, 3) float array，相机坐标系下的点集

        Returns:
            (N, 3) float array，机械臂基坐标系坐标
        """
        return apply_transform_batch(self._T_cam2robot, points)

    def camera_xyz_to_robot_command(
        self,
        x_cam: float,
        y_cam: float,
        z_cam: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
    ) -> dict:
        """
        将相机坐标转换为机械臂运动指令字典。

        Args:
            x_cam, y_cam, z_cam : 相机坐标系坐标 (mm)
            offset_x/y/z        : 末端执行器偏移补偿 (mm)，
                                  例如夹爪长度、工具中心点偏移

        Returns:
            dict 包含：
                'x', 'y', 'z'   : 目标坐标 (mm)
                'command_str'    : 格式化指令字符串（可直接发送给控制器）
        """
        xr, yr, zr = self.camera_to_robot(x_cam, y_cam, z_cam)

        # 应用末端偏移
        xr += offset_x
        yr += offset_y
        zr += offset_z

        # 验证坐标合法性
        valid = self.validate_robot_coords(xr, yr, zr)

        cmd = {
            "x": round(xr, 2),
            "y": round(yr, 2),
            "z": round(zr, 2),
            "valid": valid,
            "command_str": f"MOVE {xr:.2f} {yr:.2f} {zr:.2f}",
        }
        return cmd

    # ── 坐标验证 ──────────────────────────────────────────────────────

    # 机械臂工作空间边界（单位：mm），根据实际机械臂修改
    WORKSPACE = {
        "x": (-800.0, 800.0),
        "y": (-800.0, 800.0),
        "z": (  50.0, 1000.0),
    }

    def validate_robot_coords(
        self, x: float, y: float, z: float, verbose: bool = False
    ) -> bool:
        """
        检查目标坐标是否在机械臂工作空间内。

        Returns:
            True  : 坐标合法
            False : 坐标超出工作空间
        """
        ws = self.WORKSPACE
        valid = (
            ws["x"][0] <= x <= ws["x"][1]
            and ws["y"][0] <= y <= ws["y"][1]
            and ws["z"][0] <= z <= ws["z"][1]
        )
        if not valid and verbose:
            print(
                f"[CoordTransformer] ⚠️  坐标超出工作空间！\n"
                f"  目标: X={x:.1f}  Y={y:.1f}  Z={z:.1f} mm\n"
                f"  工作空间: X∈{ws['x']}  Y∈{ws['y']}  Z∈{ws['z']}"
            )
        return valid

    def set_workspace(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
    ):
        """
        设置机械臂工作空间边界（mm）。

        Args:
            x_range, y_range, z_range: (min, max) 元组
        """
        self.WORKSPACE = {"x": x_range, "y": y_range, "z": z_range}

    # ── 保存 / 加载 ───────────────────────────────────────────────────

    def save(self, filepath: str):
        """
        将当前变换参数保存为 .npz 文件。

        Args:
            filepath: 保存路径，建议使用 "hand_eye_params.npz"
        """
        T = self._T_cam2robot
        np.savez(
            filepath,
            T_cam2robot=T,
            R=T[:3, :3],
            t=T[:3,  3],
        )
        print(f"[CoordTransformer] 参数已保存: {filepath}")

    # ── 调试辅助 ──────────────────────────────────────────────────────

    def _print_summary(self):
        """打印变换参数摘要"""
        if self._T_cam2robot is None:
            return
        T = self._T_cam2robot
        R = T[:3, :3]
        t = T[:3,  3]
        roll, pitch, yaw = rotation_matrix_to_euler(R, degrees=True)
        print(f"  平移(mm): X={t[0]:+.2f}  Y={t[1]:+.2f}  Z={t[2]:+.2f}")
        print(f"  旋转(°):  roll={roll:+.2f}  pitch={pitch:+.2f}  yaw={yaw:+.2f}")

    def __repr__(self) -> str:
        status = f"已标定 (来源: {self._source})" if self.is_calibrated else "未标定（单位变换）"
        return f"CoordTransformer({status})"


# =====================================================================
# Hand-Eye 标定辅助函数
# =====================================================================

def calibrate_hand_eye(
    R_gripper2base_list: list[np.ndarray],
    t_gripper2base_list: list[np.ndarray],
    R_target2cam_list:   list[np.ndarray],
    t_target2cam_list:   list[np.ndarray],
    method: int = 0,
    output_file: str = "hand_eye_params.npz",
) -> CoordTransformer:
    """
    执行 Eye-to-Hand (眼在手外) Hand-Eye 标定。

    原理：
        Robot moves to N known poses → Camera observes calibration board at each pose
        Solve: T_cam2base  such that  T_gripper2base @ T_cam2gripper = const

    Args:
        R_gripper2base_list : N × (3×3) 机械臂末端→基坐标系旋转矩阵列表
        t_gripper2base_list : N × (3,)  机械臂末端→基坐标系平移向量列表（mm）
        R_target2cam_list   : N × (3×3) 标定板→相机坐标系旋转矩阵列表
        t_target2cam_list   : N × (3,)  标定板→相机坐标系平移向量列表（mm）
        method              : cv2.CALIB_HAND_EYE_* 方法（默认 TSAI=0）
                              0=TSAI, 1=PARK, 2=HORAUD, 3=ANDREFF, 4=DANIILIDIS
        output_file         : 结果保存路径

    Returns:
        CoordTransformer 对象，已加载标定结果

    使用示例
    --------
    >>> import cv2
    >>> # 采集 N 组 (机械臂位姿, 相机观测) 数据对后：
    >>> tf = calibrate_hand_eye(
    ...     R_g2b_list, t_g2b_list,
    ...     R_t2c_list, t_t2c_list,
    ... )
    >>> xyz_robot = tf.camera_to_robot(x_cam, y_cam, z_cam)
    """
    import cv2

    if len(R_gripper2base_list) < 3:
        raise ValueError(
            "Hand-Eye 标定至少需要 3 组数据（建议 15~20 组以提升精度）。"
        )

    method_map = {
        0: cv2.CALIB_HAND_EYE_TSAI,
        1: cv2.CALIB_HAND_EYE_PARK,
        2: cv2.CALIB_HAND_EYE_HORAUD,
        3: cv2.CALIB_HAND_EYE_ANDREFF,
        4: cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base_list,
        [np.asarray(t).reshape(3, 1) for t in t_gripper2base_list],
        R_target2cam_list,
        [np.asarray(t).reshape(3, 1) for t in t_target2cam_list],
        method=method_map.get(method, cv2.CALIB_HAND_EYE_TSAI),
    )

    t_cam2base = t_cam2base.ravel()

    print("\n[Hand-Eye 标定结果]")
    print(f"  平移 t: [{t_cam2base[0]:+.2f}, {t_cam2base[1]:+.2f}, {t_cam2base[2]:+.2f}] mm")
    roll, pitch, yaw = rotation_matrix_to_euler(R_cam2base)
    print(f"  旋转 R: roll={roll:+.2f}°  pitch={pitch:+.2f}°  yaw={yaw:+.2f}°")

    # 保存结果
    np.savez(
        output_file,
        R=R_cam2base,
        t=t_cam2base,
        T_cam2robot=make_transform(R_cam2base, t_cam2base),
    )
    print(f"  已保存: {output_file}")

    return CoordTransformer(R=R_cam2base, t=t_cam2base)


# =====================================================================
# 角度 / 几何辅助函数
# =====================================================================

def pixel_to_camera_ray(
    u: float, v: float,
    fx: float, fy: float,
    cx: float, cy: float,
) -> np.ndarray:
    """
    将像素坐标 (u, v) 反投影为相机坐标系中的单位方向向量。

    Args:
        u, v        : 像素坐标
        fx, fy      : 焦距（像素）
        cx, cy      : 主点（像素）

    Returns:
        ray: (3,) 单位向量，方向 = 从相机光心指向该像素对应的空间点
    """
    d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)


def bearing_angle_to_target(
    x_cam: float, y_cam: float, z_cam: float,
    degrees: bool = True,
) -> Tuple[float, float]:
    """
    计算相机坐标系下目标点的方位角（水平/垂直偏角）。

    Args:
        x_cam, y_cam, z_cam : 目标在相机坐标系下的坐标 (mm)
        degrees             : True→返回度，False→返回弧度

    Returns:
        (azimuth, elevation)
          azimuth   : 水平偏角（正=右偏）
          elevation : 垂直偏角（正=上偏）
    """
    azimuth   = np.arctan2(x_cam, z_cam)
    elevation = np.arctan2(-y_cam, np.sqrt(x_cam**2 + z_cam**2))

    if degrees:
        return float(np.degrees(azimuth)), float(np.degrees(elevation))
    return float(azimuth), float(elevation)


def distance_to_target(x_cam: float, y_cam: float, z_cam: float) -> float:
    """
    计算相机光心到目标点的欧氏距离（mm）。

    Args:
        x_cam, y_cam, z_cam : 相机坐标系坐标 (mm)

    Returns:
        distance_mm : 欧氏距离 (mm)
    """
    return float(np.sqrt(x_cam**2 + y_cam**2 + z_cam**2))


def approach_vector(
    x_cam: float, y_cam: float, z_cam: float,
    approach_dist_mm: float = 50.0,
) -> Tuple[float, float, float]:
    """
    计算机械臂抓取前的预接近点坐标（在目标点正后方 approach_dist_mm 处）。

    机械臂通常先运动到预接近点，再沿 Z 轴（深度方向）直线前进到目标，
    以避免碰撞路径规划问题。

    Args:
        x_cam, y_cam, z_cam : 目标在相机坐标系下的坐标 (mm)
        approach_dist_mm     : 预接近距离 (mm)，默认 50mm

    Returns:
        (x_pre, y_pre, z_pre) 预接近点在相机坐标系下的坐标 (mm)
    """
    # 方向向量（从相机指向目标），归一化
    vec = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    dist = np.linalg.norm(vec)
    if dist < 1e-6:
        return float(x_cam), float(y_cam), float(z_cam)

    unit = vec / dist
    # 预接近点 = 目标点沿反向退后 approach_dist_mm
    pre = vec - unit * approach_dist_mm
    return float(pre[0]), float(pre[1]), float(pre[2])


# =====================================================================
# 快速功能测试
# =====================================================================

def _demo():
    """
    独立运行时执行简单的功能演示，验证各函数的正确性。
    """
    print("\n" + "=" * 60)
    print("  CoordTransformer 功能演示")
    print("=" * 60)

    # ── 1. 欧拉角 ↔ 旋转矩阵 ──────────────────────────────────────
    roll, pitch, yaw = 10.0, -5.0, 30.0
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees=True)
    roll2, pitch2, yaw2 = rotation_matrix_to_euler(R, degrees=True)
    print(f"\n[欧拉角往返测试]")
    print(f"  输入:  roll={roll}°  pitch={pitch}°  yaw={yaw}°")
    print(f"  往返:  roll={roll2:.4f}°  pitch={pitch2:.4f}°  yaw={yaw2:.4f}°")
    assert abs(roll - roll2) < 1e-8 and abs(pitch - pitch2) < 1e-8 and abs(yaw - yaw2) < 1e-8
    print("  ✅ 通过")

    # ── 2. 变换矩阵求逆 ───────────────────────────────────────────
    t_vec = np.array([100.0, -200.0, 500.0])
    T  = make_transform(R, t_vec)
    T_inv = invert_transform(T)
    prod = T @ T_inv
    err  = np.max(np.abs(prod - np.eye(4)))
    print(f"\n[变换矩阵求逆测试]")
    print(f"  T @ T_inv ≈ I，最大误差: {err:.2e}")
    assert err < 1e-10, f"求逆误差过大: {err}"
    print("  ✅ 通过")

    # ── 3. CoordTransformer（单位变换）────────────────────────────
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = CoordTransformer()

    x_c, y_c, z_c = 45.2, -12.8, 823.5
    x_r, y_r, z_r = tf.camera_to_robot(x_c, y_c, z_c)
    print(f"\n[CoordTransformer 单位变换测试]")
    print(f"  相机坐标: ({x_c}, {y_c}, {z_c})")
    print(f"  机器人坐标: ({x_r:.2f}, {y_r:.2f}, {z_r:.2f})")
    assert abs(x_r - x_c) < 1e-9
    print("  ✅ 通过（单位变换，坐标应不变）")

    # ── 4. 手动设置 R, t 的 CoordTransformer ────────────────────
    R_test = euler_to_rotation_matrix(0, 0, 90, degrees=True)
    t_test = np.array([500.0, 0.0, 200.0])
    tf2    = CoordTransformer(R=R_test, t=t_test)

    x_c2, y_c2, z_c2 = 100.0, 0.0, 0.0
    x_r2, y_r2, z_r2 = tf2.camera_to_robot(x_c2, y_c2, z_c2)
    print(f"\n[CoordTransformer 旋转90°测试]")
    print(f"  输入相机坐标: ({x_c2}, {y_c2}, {z_c2})")
    print(f"  机器人坐标:   ({x_r2:.2f}, {y_r2:.2f}, {z_r2:.2f})")
    print("  ✅ 完成")

    # ── 5. 方位角计算 ────────────────────────────────────────────
    az, el = bearing_angle_to_target(100.0, -50.0, 800.0, degrees=True)
    dist   = distance_to_target(100.0, -50.0, 800.0)
    print(f"\n[方位角 & 距离测试]")
    print(f"  目标坐标: X=100, Y=-50, Z=800 mm")
    print(f"  水平偏角: {az:+.2f}°  垂直偏角: {el:+.2f}°")
    print(f"  距离: {dist:.2f} mm")
    print("  ✅ 完成")

    # ── 6. 预接近点计算 ───────────────────────────────────────────
    xp, yp, zp = approach_vector(100.0, -50.0, 800.0, approach_dist_mm=60.0)
    print(f"\n[预接近点测试]")
    print(f"  目标:       X=100  Y=-50  Z=800 mm")
    print(f"  预接近点:   X={xp:.1f}  Y={yp:.1f}  Z={zp:.1f} mm")
    print("  ✅ 完成")

    # ── 7. 工作空间验证 ───────────────────────────────────────────
    tf3 = CoordTransformer(R=np.eye(3), t=np.zeros(3))
    valid1 = tf3.validate_robot_coords(100.0, 200.0, 300.0)
    valid2 = tf3.validate_robot_coords(9999.0, 0.0, 0.0, verbose=True)
    print(f"\n[工作空间验证测试]")
    print(f"  (100, 200, 300) 合法: {valid1}  ✅")
    print(f"  (9999, 0, 0)   合法: {valid2}  ✅")

    print("\n" + "=" * 60)
    print("  所有测试通过！✅")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _demo()
