# 基于 YOLO 分割的气瓶自动标注项目

[English README](./README.md)

这个仓库整理了我的毕业设计相关代码，内容包括气瓶图像自动标注、数据集整理、模型训练、Web 端推理演示，以及后续的定位实验代码。

## 核心自动标注思路

这个项目的核心不是直接训练一个专门的气瓶分割模型来做初标，而是先利用通用 YOLO 分割模型进行低阈值检测，再结合气瓶场景规则，把与气瓶形态相近的目标筛选出来。

主要步骤如下：

1. 使用很低的置信度阈值（`0.01`）运行 YOLO 分割。
2. 只保留那些经常会被误识别成气瓶的 COCO 类别，例如 `bottle`、`fire hydrant`、`sink`、`toilet`、`cup`、`chair`、`vase`。
3. 过滤掉面积太小或离图像中心太远的候选目标。
4. 对剩余候选目标计算综合分数：

```text
score = confidence * class_weight * centrality_bonus
```

5. 选出得分最高的多边形轮廓，导出为 YOLO 分割标注，或者叠加到可视化结果图中。

## `process_E_to_F.py` 为什么重要

[process_E_to_F.py](./process_E_to_F.py) 是这套鲁棒自动标注流程里最清晰的批处理版本之一，基本代表了项目的核心自动标注实现。

它主要做了这些事：

- 加载 `yolo11x-seg.pt`
- 如果检测到 CUDA，则自动优先使用 GPU
- 使用 `numpy + cv2.imdecode` 读取图片，适合 Windows 和中文路径场景
- 复用 `GasCylinderSegmenter` 的核心筛选逻辑
- 把图片从一个目录批量读入，再把处理结果输出到另一个目录

在原始本地环境里，这个脚本默认使用：

- 输入目录：`E:\gas`
- 输出目录：`F:\guoji`

如果换机器运行，需要先把这些路径改成你自己的本地路径。

## 核心算法代码在哪些文件里

同一套自动标注核心逻辑，主要出现在这些文件中：

- [process_E_to_F.py](./process_E_to_F.py)
- [segmentation_engine.py](./segmentation_engine.py)
- [auto_label_final_robust.py](./auto_label_final_robust.py)
- [app_v2.py](./app_v2.py)
- [batch_process_gpu_v3.py](./batch_process_gpu_v3.py)

其中：

- `process_E_to_F.py` 更偏向批量处理入口
- `segmentation_engine.py` 更适合作为可复用的核心引擎
- `auto_label_final_robust.py` 会直接输出 YOLO 分割标签
- `app_v2.py` 提供了一个基于 FastAPI 的网页演示版本

## 仓库结构说明

- `process_E_to_F.py`：批量推理和结果导出脚本
- `segmentation_engine.py`：可复用的分割引擎
- `app_v2.py`：FastAPI Web 推理演示
- `auto_label_final_robust.py`：鲁棒自动标注脚本，输出 YOLO 分割标签
- `auto_segment_*.py`、`batch_process_gpu*.py`：自动标注流程的不同实验版本
- `prepare_dataset.py`、`visualize_dataset.py`、`train_yolo.py`：本地数据准备和训练辅助脚本
- `colab_auto_label.py`、`colab_train.py`、`colab_dataset_clean.py`：Google Colab 工作流的 Python 导出版本
- `colab_*.ipynb`：Colab 笔记本版本
- `localization/`：双目、三维定位和坐标变换实验代码

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行前说明

- 很多脚本仍然保留了原始本地 Windows 路径，例如 `D:\biyesheji\...`、`E:\gas`、`F:\guoji`。
- 在别的电脑上运行前，需要先改成自己的路径。
- 模型权重（`*.pt`）、本地数据集、可视化结果图、调试日志和论文材料没有提交到 GitHub。
- 当前仓库主要公开的是源码和流程，便于别人研究自动标注方法本身。
- [data.yaml](./data.yaml) 只是一个轻量示例，默认假设本地存在 `gas_dataset` 目录。
