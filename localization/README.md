# 🎯 3D 精确定位系统 | 3D Object Localization System

> 毕业设计第二阶段：在 YOLO 检测基础上，利用双目/多目视觉实现气瓶的精确三维定位，为机械臂操作提供坐标输入。

---

## 📂 文件结构

```
localization/
├── README.md                        ← 本文件，总体说明
│
├── 【方案一】双目立体视觉（推荐）
│   ├── stereo_calibration.py        ← Step 1: 双目相机标定（棋盘格）
│   ├── stereo_3d_localize.py        ← Step 2: YOLO检测 + SGBM视差 → 3D坐标
│   └── stereo_test_image.py         ← Step 3: 静态图片测试
│
├── 【方案二】单目深度估计（无需双目）
│   ├── depth_anything_localize.py   ← Depth Anything V2 + YOLO → 相对深度
│   └── depth_realtime.py            ← 实时深度可视化
│
├── 【方案三】多摄像头三角测量（最精准）
│   ├── multicam_calibration.py      ← 多摄像头标定（含基础矩阵F）
│   └── triangulation_3d.py          ← YOLO + 极线几何三角测量 → 3D坐标
│
├── utils/
│   ├── draw_utils.py                ← 3D坐标可视化工具
│   └── coord_transform.py           ← 相机坐标系 → 机械臂坐标系转换
│
└── requirements.txt                 ← 依赖库
```

---

## 🔬 技术方案对比

| 方案 | 技术核心 | 硬件需求 | 精度 | 难度 | 适用场景 |
|------|---------|----------|------|------|---------|
| **方案一：双目立体视觉** | OpenCV SGBM + 三角投影 | 2个普通USB相机（固定基线） | ★★★★☆ | ★★☆☆☆ | **推荐！毕设首选** |
| **方案二：单目深度估计** | Depth Anything V2 (NeurIPS 2024) | 1个相机 | ★★★☆☆（相对深度）| ★☆☆☆☆ | 无双目相机时的备选 |
| **方案三：多摄像头三角测量** | 极线几何 + RANSAC三角化 | 2+个相机（任意摆放） | ★★★★★ | ★★★★☆ | 精度要求极高时 |

---

## 🚀 快速开始

### 安装依赖

```bash
cd localization
pip install -r requirements.txt
```

### 方案一：双目立体视觉（推荐路线）

**硬件准备：**
- 2 个相同型号的 USB 摄像头（建议间距 6~15cm，即基线距离）
- 打印一张 9×6 的棋盘格标定板（格子边长建议 2.5cm）

**流程：**

```
Step 1: 标定
python stereo_calibration.py
→ 采集30张以上棋盘格图片 → 生成 stereo_params.npz

Step 2: 实时定位
python stereo_3d_localize.py
→ 实时显示：YOLO检测框 + 气瓶3D坐标 (X, Y, Z mm)

Step 3: 静态测试
python stereo_test_image.py --left left.jpg --right right.jpg
```

**输出坐标示例：**
```
[气瓶检测] 置信度: 0.94
  相机坐标系: X=+45.2mm  Y=-12.8mm  Z=+823.5mm
  机械臂坐标系: X=+823.5  Y=-45.2  Z=+12.8  (mm)
```

---

### 方案二：Depth Anything V2（单摄像头）

**特点：** 不需要双目相机，直接从单张图片估计深度，为 NeurIPS 2024 最新成果（7600+ Stars）。

**注意：** 输出的是相对深度（0~1 归一化），需要用已知尺寸物体进行比例标定才能得到真实距离。

```bash
# 首次运行会自动下载模型（约 100MB）
python depth_anything_localize.py
```

---

### 方案三：多摄像头三角测量（高精度）

**特点：** 基于极线几何（Epipolar Geometry），原理严谨，适合精度要求高的场合。来源参考：`labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking`

```bash
# Step 1: 标定（需要特征点对应关系）
python multicam_calibration.py

# Step 2: 运行三角测量
python triangulation_3d.py
```

---

## 📐 核心原理简介

### 双目视觉测距原理

```
左相机 CL          右相机 CR
    \                  /
     \   基线 B       /
      \              /
       \            /
        \          /
         \        /
          目标物体 P

深度 Z = (f × B) / d
  f = 焦距（像素）
  B = 基线距离（mm）
  d = 视差 = x_left - x_right（像素差）
```

### 完整流程

```
双目图像输入
    ↓
[YOLO11检测] → 找到气瓶的像素坐标 (u, v)
    ↓
[双目校正] → 消除畸变，对齐极线
    ↓
[SGBM视差计算] → 生成视差图 D(u,v)
    ↓
[3D重投影] → (X, Y, Z) = Q × (u, v, d, 1)T
    ↓
[坐标变换] → 相机坐标系 → 机械臂基坐标系
    ↓
输出：气瓶中心三维坐标 (X, Y, Z) mm
```

---

## 🔧 参数说明

### 双目标定参数（stereo_params.npz 中保存）

| 参数 | 说明 |
|------|------|
| `mtx_left / mtx_right` | 左/右相机内参矩阵 (3×3) |
| `dist_left / dist_right` | 左/右相机畸变系数 |
| `R` | 两相机间旋转矩阵 |
| `T` | 两相机间平移向量（即基线） |
| `Q` | 视差→深度重投影矩阵 (4×4) |

### SGBM 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `numDisparities` | 64~128 | 搜索视差范围，需为16的倍数 |
| `blockSize` | 5~11 | 匹配块大小，奇数 |
| `minDisparity` | 0 | 最小视差 |
| `uniquenessRatio` | 5~15 | 唯一性过滤 |

---

## 💡 与机械臂集成思路

```python
# 将3D坐标发送给机械臂控制器（示例）
import serial  # 或 socket

def send_to_robot_arm(x_mm, y_mm, z_mm):
    """
    将气瓶中心坐标发送给机械臂
    坐标已转换到机械臂基坐标系
    """
    command = f"MOVE {x_mm:.1f} {y_mm:.1f} {z_mm:.1f}\n"
    # ser.write(command.encode())  # 串口发送
    print(f"[机械臂指令] {command}")
```

---

## 📚 参考项目与论文

| 项目/论文 | 链接 | 说明 |
|-----------|------|------|
| **Depth Anything V2** (NeurIPS 2024) | [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | 最强单目深度估计，7600+ Stars |
| **labvisio/Multi-Object-Triangulation** | [GitHub](https://github.com/labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking) | 多摄像头YOLO三角测量 |
| **niconielsen32/YOLO-3D** | [GitHub](https://github.com/niconielsen32/YOLO-3D) | YOLO+深度 3D检测 387 Stars |
| **FoundationPose** (CVPR 2024 Highlight) | [GitHub](https://github.com/NVlabs/FoundationPose) | NVIDIA 6DoF姿态估计 1.1k Stars |
| **TemugeB/python_stereo_camera_calibrate** | [GitHub](https://github.com/TemugeB/python_stereo_camera_calibrate) | 双目标定参考实现 258 Stars |
| **YOLO-ICP** (IEEE Sensors 2024) | Fraunhofer IKTS | YOLO+点云配准，6DoF Bin-Picking |
| **stereo_vision distance estimation** | Springer 2024 | YOLO+MLP测距，精度98.15% |

---

## ⚡ 硬件推荐（按预算排序）

| 方案 | 设备 | 优点 | 参考价格 |
|------|------|------|---------|
| **入门** | 2× 普通 USB 摄像头（固定在同一支架） | 便宜易得 | ¥50~100 |
| **进阶** | Intel RealSense D435i（RGB-D相机） | 自带深度，免标定 | ¥1500~2000 |
| **专业** | ZED 2 立体相机（Stereolabs） | 高精度，SDK完善 | ¥4000+ |

---

## 🐛 常见问题

**Q: 标定失败，找不到棋盘格角点？**
> 确保棋盘格打印清晰无反光，拍摄时覆盖画面各个角落，至少采集 20 张以上。

**Q: 视差图噪声很大，深度不准？**
> 调大 `blockSize`（但会损失空间分辨率），同时确保光照均匀，避免强反光表面。

**Q: Depth Anything V2 模型下载失败？**
> 使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com`，然后重新运行脚本。

**Q: 气瓶是圆柱体，中心点难以稳定检测？**
> 使用 YOLO 分割模型（`yolo11x-seg.pt`，项目中已有），取 mask 的质心作为目标点，比 bounding box 中心更准确。

---

*本模块为毕业设计"基于视觉的气瓶识别与机械臂定位系统"第二阶段，建立在第一阶段 YOLO 训练基础之上。*