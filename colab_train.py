# ============================================================
#  气瓶识别 YOLO 模型训练 — Google Colab Training Pipeline
#
#  流程:
#  1. 安装依赖
#  2. HuggingFace 认证
#  3. 下载数据集（train + val）
#  4. 验证数据集
#  5. 配置训练参数
#  6. 开始训练
#  7. 查看训练结果
#  8. 上传模型到 HuggingFace
#
#  使用方法:
#  1. 打开 https://colab.research.google.com
#  2. 上传此文件：文件 → 上传笔记本 → 选择此 .py 文件
#  3. 启用 GPU：运行时 → 更改运行时类型 → T4 GPU
#  4. 从上到下依次运行每个单元格
# ============================================================


# %% [markdown]
# # 🚀 气瓶识别 YOLO 模型训练
#
# 本笔记本完成从数据集下载到模型上传的完整训练流程：
#
# | 步骤 | 内容 |
# |------|------|
# | Step 1 | 安装依赖 |
# | Step 2 | HuggingFace 认证 |
# | Step 3 | 下载完整数据集 |
# | Step 4 | 验证数据集完整性 |
# | Step 5 | 配置训练参数 |
# | Step 6 | 开始训练 |
# | Step 7 | 查看训练结果 |
# | Step 8 | 上传模型到 HuggingFace |
#
# **⚠️ 运行前请确认已开启 GPU：运行时 → 更改运行时类型 → T4 GPU**


# %% [markdown]
# ---
# ## ⚙️ Step 1 — 安装依赖 | Install Dependencies


# %%
# @title ⚙️ Step 1 — 安装依赖 | Install Dependencies

get_ipython().system('pip install ultralytics huggingface_hub -q')

import torch, os, sys
from pathlib import Path

print('✅ 依赖安装完成')
print(f'   Python      : {sys.version.split()[0]}')
print(f'   PyTorch     : {torch.__version__}')
print(f'   CUDA 可用   : {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'   GPU 型号    : {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'   显存大小    : {total_mem:.1f} GB')
else:
    print()
    print('❌ 未检测到 GPU！')
    print('   请前往：运行时 → 更改运行时类型 → T4 GPU → 保存')
    print('   然后重新运行所有单元格')


# %% [markdown]
# ---
# ## 🔑 Step 2 — HuggingFace 认证 | Authentication


# %%
# @title 🔑 Step 2 — HuggingFace 认证 | Authentication
# @markdown 填入 HuggingFace Token（需要 **write** 权限）
# @markdown
# @markdown 获取地址：https://huggingface.co/settings/tokens → New token → Role: Write

HF_TOKEN     = ''              # @param {type:"string"}
DATASET_REPO = 'rick003/11111' # @param {type:"string"}

from huggingface_hub import login, HfApi

if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('请输入 HuggingFace Token（输入内容不可见）: ')

if not HF_TOKEN:
    raise ValueError('❌ 必须提供 HuggingFace Token 才能继续')

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

try:
    info = api.dataset_info(DATASET_REPO)
    print(f'✅ HuggingFace 认证成功')
    print(f'   数据集 : {DATASET_REPO}')
    print(f'   地址   : https://huggingface.co/datasets/{DATASET_REPO}')
except Exception as e:
    print(f'❌ 数据集访问失败: {e}')
    raise


# %% [markdown]
# ---
# ## 📥 Step 3 — 下载数据集 | Download Dataset
#
# 使用 `snapshot_download` 一次性下载完整数据集（train + val），
# 约 **7.6 GB**，T4 Colab 磁盘空间充足（约 100 GB）。
# 下载时间取决于网络速度，一般 **10~30 分钟**。


# %%
# @title 📥 Step 3 — 下载数据集 | Download Dataset
# @markdown 下载完整数据集到 Colab 本地（约 7.6 GB，请耐心等待）
# @markdown
# @markdown 已下载过的文件会自动跳过（断点续传）

from huggingface_hub import snapshot_download
import time

DATASET_LOCAL = '/content/gas_dataset'

print(f'📥 开始下载数据集: {DATASET_REPO}')
print(f'   本地路径: {DATASET_LOCAL}')
print(f'   跳过文件: preview/, runs/, *.cache')
print(f'   预计大小: ~7.6 GB\n')

t0 = time.time()

try:
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type='dataset',
        token=HF_TOKEN,
        local_dir=DATASET_LOCAL,
        ignore_patterns=[
            'preview/**',
            'runs/**',
            '*.cache',
            '.gitattributes',
        ],
    )
except TypeError:
    # 兼容旧版 huggingface_hub
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type='dataset',
        token=HF_TOKEN,
        local_dir=DATASET_LOCAL,
        ignore_patterns=[
            'preview/**',
            'runs/**',
            '*.cache',
            '.gitattributes',
        ],
    )

elapsed = time.time() - t0
print(f'\n✅ 下载完成！耗时 {elapsed/60:.1f} 分钟')

# 统计文件数量
base = Path(DATASET_LOCAL)
train_imgs  = list((base / 'images/train').glob('*.png')) + \
              list((base / 'images/train').glob('*.jpg'))
val_imgs    = list((base / 'images/val').glob('*.png')) + \
              list((base / 'images/val').glob('*.jpg'))
train_lbls  = list((base / 'labels/train').glob('*.txt'))
val_lbls    = list((base / 'labels/val').glob('*.txt'))

SEP = '=' * 50
print(f'\n{SEP}')
print('📊 数据集统计:')
print(f'   训练图像 : {len(train_imgs):>5} 张')
print(f'   训练标签 : {len(train_lbls):>5} 个')
print(f'   验证图像 : {len(val_imgs):>5} 张')
print(f'   验证标签 : {len(val_lbls):>5} 个')
print(f'{SEP}')


# %% [markdown]
# ---
# ## 🔍 Step 4 — 验证数据集 | Validate Dataset


# %%
# @title 🔍 Step 4 — 验证数据集 | Validate Dataset
# @markdown 检查图像与标签的对应关系，确认格式正确，生成 dataset.yaml

import yaml, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2, numpy as np

base = Path(DATASET_LOCAL)

# ── 检查图像与标签是否一一对应 ─────────────────────────────────────────────
def check_split(split_name):
    img_dir = base / f'images/{split_name}'
    lbl_dir = base / f'labels/{split_name}'

    imgs = {p.stem for p in img_dir.glob('*')
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
    lbls = {p.stem for p in lbl_dir.glob('*.txt')}

    no_label  = imgs - lbls
    no_image  = lbls - imgs
    empty_lbl = [p for p in lbl_dir.glob('*.txt') if p.stat().st_size == 0]

    print(f'\n【{split_name}】')
    print(f'   图像数   : {len(imgs)}')
    print(f'   标签数   : {len(lbls)}')
    print(f'   缺标签   : {len(no_label)} 张  {list(no_label)[:3] if no_label else ""}')
    print(f'   缺图像   : {len(no_image)} 个  {list(no_image)[:3] if no_image else ""}')
    print(f'   空标签   : {len(empty_lbl)} 个')

    return len(imgs), len(lbls)

print('🔍 检查数据集完整性...')
n_train_imgs, n_train_lbls = check_split('train')
n_val_imgs,   n_val_lbls   = check_split('val')

# ── 写入 dataset.yaml ──────────────────────────────────────────────────────
YAML_PATH = f'{DATASET_LOCAL}/dataset.yaml'
yaml_content = {
    'path' : DATASET_LOCAL,
    'train': 'images/train',
    'val'  : 'images/val',
    'names': {0: 'gas_cylinder'},
}
with open(YAML_PATH, 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

print(f'\n✅ dataset.yaml 已写入: {YAML_PATH}')
print(f'   内容:')
with open(YAML_PATH) as f:
    for line in f:
        print(f'      {line}', end='')

# ── 可视化随机抽查 3 张标注 ─────────────────────────────────────────────────
print('\n\n🖼️  随机抽查 3 张训练图像的标注...')

sample_imgs = random.sample(
    list((base / 'images/train').glob('*.png')) +
    list((base / 'images/train').glob('*.jpg')),
    min(3, n_train_imgs)
)

fig, axes = plt.subplots(1, len(sample_imgs), figsize=(len(sample_imgs) * 7, 5))
if len(sample_imgs) == 1:
    axes = [axes]

for ax, img_path in zip(axes, sample_imgs):
    raw = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 缩放用于显示
    scale = 800 / w
    disp  = cv2.resize(img, (800, int(h * scale)), interpolation=cv2.INTER_AREA)
    dh, dw = disp.shape[:2]

    lbl_path = base / 'labels/train' / (img_path.stem + '.txt')
    if lbl_path.exists() and lbl_path.stat().st_size > 0:
        content = lbl_path.read_text().strip()
        parts   = content.split()
        if len(parts) >= 7:
            vals = [float(v) for v in parts[1:]]
            pts  = (np.array(vals).reshape(-1, 2) * [dw, dh]).astype(np.int32).reshape(-1, 1, 2)
            ov   = disp.copy()
            cv2.fillPoly(ov, [pts], (0, 255, 0))
            cv2.addWeighted(ov, 0.35, disp, 0.65, 0, disp)
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
            n_pts = (len(parts) - 1) // 2
            title = f'{img_path.name}\n{n_pts} pts'
        else:
            title = f'{img_path.name}\n(标签格式异常)'
    else:
        title = f'{img_path.name}\n(无标签)'

    ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()
print('✅ 数据集验证完成，可以开始训练！')


# %% [markdown]
# ---
# ## ⚙️ Step 5 — 配置训练参数 | Training Configuration


# %%
# @title ⚙️ Step 5 — 配置训练参数 | Training Configuration
# @markdown ### 模型选择
# @markdown | 模型 | 速度 | 精度 | 适用场景 |
# @markdown |------|------|------|----------|
# @markdown | yolo11n-seg | ⚡⚡⚡ | ⭐⭐ | 快速验证 |
# @markdown | yolo11m-seg | ⚡⚡ | ⭐⭐⭐ | 平衡 |
# @markdown | yolo11x-seg | ⚡ | ⭐⭐⭐⭐⭐ | 最终训练（推荐） |
# @markdown
# @markdown ### 参数说明
# @markdown - **EPOCHS**: 训练轮数，建议 100~200
# @markdown - **PATIENCE**: 早停轮数，连续 N 轮无提升则自动停止
# @markdown - **IMG_SIZE**: 输入图像尺寸，640 是 YOLO 标准
# @markdown - **BATCH**: -1 = 自动选择（推荐），或手动填 8 / 16

MODEL    = 'yolo11x-seg.pt'  # @param ["yolo11n-seg.pt","yolo11m-seg.pt","yolo11x-seg.pt"]
EPOCHS   = 100               # @param {type:"integer"}
IMG_SIZE = 640               # @param {type:"integer"}
BATCH    = -1                # @param {type:"integer"}
PATIENCE = 30                # @param {type:"integer"}
RUN_NAME = 'gas_seg_v1'      # @param {type:"string"}

# 检查 GPU 显存，给出建议
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb < 8 and BATCH == -1:
        print(f'⚠️  检测到显存 {vram_gb:.1f} GB，建议将 BATCH 手动设为 4 或 8')
    elif vram_gb >= 15:
        print(f'✅ 显存充足 ({vram_gb:.1f} GB)，BATCH=-1 自动模式可以正常使用')

RUN_DIR = f'/content/runs/{RUN_NAME}'

SEP = '=' * 50
print(f'\n{SEP}')
print('📋 训练配置:')
print(f'   模型        : {MODEL}')
print(f'   训练轮数    : {EPOCHS}')
print(f'   图像尺寸    : {IMG_SIZE}')
print(f'   批次大小    : {"自动" if BATCH == -1 else BATCH}')
print(f'   早停轮数    : {PATIENCE}')
print(f'   运行名称    : {RUN_NAME}')
print(f'   输出目录    : {RUN_DIR}')
print(f'   数据集配置  : {YAML_PATH}')
print(f'{SEP}')


# %% [markdown]
# ---
# ## 🚀 Step 6 — 开始训练 | Start Training
#
# ⏱️ **预计时间参考（T4 GPU，~280 张图，100 epochs）：**
#
# | 模型 | 预计时间 |
# |------|----------|
# | yolo11n-seg | ~20 分钟 |
# | yolo11m-seg | ~45 分钟 |
# | yolo11x-seg | ~90 分钟 |
#
# 训练过程中可以看到每个 epoch 的 loss 和 mAP 实时输出。


# %%
# @title 🚀 Step 6 — 开始训练 | Start Training
# @markdown 运行后会实时打印每轮的 loss / mAP，训练结束自动保存最优模型

from ultralytics import YOLO
import time

print(f'🚀 加载预训练模型: {MODEL}')
model = YOLO(MODEL)

print(f'🏋️  开始训练...\n')
t_start = time.time()

train_results = model.train(
    data      = YAML_PATH,
    epochs    = EPOCHS,
    imgsz     = IMG_SIZE,
    batch     = BATCH,
    patience  = PATIENCE,
    project   = '/content/runs',
    name      = RUN_NAME,
    device    = '0' if torch.cuda.is_available() else 'cpu',
    workers   = 2,
    amp       = True,       # 自动混合精度，节省显存
    plots     = True,       # 生成训练曲线图
    save      = True,
    verbose   = True,
    # 数据增强（针对气瓶场景适当增强）
    hsv_h     = 0.015,
    hsv_s     = 0.7,
    hsv_v     = 0.4,
    degrees   = 5.0,        # 轻微旋转（气瓶通常竖立，不宜大幅旋转）
    translate = 0.1,
    scale     = 0.5,
    flipud    = 0.0,        # 不上下翻转（气瓶有方向性）
    fliplr    = 0.5,        # 左右翻转
    mosaic    = 0.8,
    copy_paste= 0.1,
)

t_total = time.time() - t_start

# ── 找到实际输出目录（YOLO 可能自动给名称加编号）──────────────────────────
actual_run_dir = Path(train_results.save_dir)
RUN_DIR = str(actual_run_dir)

SEP = '=' * 50
print(f'\n{SEP}')
print(f'🎉 训练完成！总耗时: {t_total/60:.1f} 分钟')
print(f'   输出目录    : {RUN_DIR}')
print(f'   最优模型    : {RUN_DIR}/weights/best.pt')
print(f'   最终模型    : {RUN_DIR}/weights/last.pt')
print(f'{SEP}')


# %% [markdown]
# ---
# ## 📊 Step 7 — 查看训练结果 | View Results


# %%
# @title 📊 Step 7 — 查看训练结果 | View Results

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path

run_path = Path(RUN_DIR)

SEP = '=' * 50

# ── 读取训练指标 ─────────────────────────────────────────────────────────────
results_csv = run_path / 'results.csv'
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    # 找到最佳 epoch
    map_col = [c for c in df.columns if 'mAP50-95' in c or 'mAP50' in c]
    if map_col:
        best_idx = df[map_col[0]].idxmax()
        best_epoch = int(df['epoch'].iloc[best_idx]) if 'epoch' in df.columns else best_idx + 1
        best_map50 = None
        best_map50_95 = None

        for col in df.columns:
            if 'mAP50-95' in col:
                best_map50_95 = df[col].iloc[best_idx]
            elif 'mAP50' in col:
                best_map50 = df[col].iloc[best_idx]

        print(f'\n{SEP}')
        print('📈 最优结果:')
        print(f'   最优 Epoch      : {best_epoch} / {len(df)}')
        if best_map50 is not None:
            print(f'   mAP50           : {best_map50:.4f}  ({best_map50*100:.1f}%)')
        if best_map50_95 is not None:
            print(f'   mAP50-95        : {best_map50_95:.4f}  ({best_map50_95*100:.1f}%)')
        print(f'{SEP}')

        # 评估达标情况（目标：总体识别准确率 ≥ 90%）
        if best_map50 is not None:
            if best_map50 >= 0.90:
                print(f'✅ mAP50 = {best_map50*100:.1f}% ≥ 90%，满足毕设要求！')
            else:
                print(f'⚠️  mAP50 = {best_map50*100:.1f}% < 90%，建议增加数据或延长训练轮数')

# ── 显示训练曲线 ─────────────────────────────────────────────────────────────
print('\n📈 训练曲线:')
results_png = run_path / 'results.png'
if results_png.exists():
    img = mpimg.imread(str(results_png))
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training Results', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print('   results.png 未找到')

# ── 显示混淆矩阵 ─────────────────────────────────────────────────────────────
conf_matrix = run_path / 'confusion_matrix_normalized.png'
if not conf_matrix.exists():
    conf_matrix = run_path / 'confusion_matrix.png'
if conf_matrix.exists():
    print('\n📊 混淆矩阵:')
    img = mpimg.imread(str(conf_matrix))
    plt.figure(figsize=(6, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ── 在验证集上跑推理，展示预测样例 ───────────────────────────────────────────
print('\n🖼️  验证集预测样例（随机 4 张）:')
best_model = YOLO(f'{RUN_DIR}/weights/best.pt')

val_images = (
    list((Path(DATASET_LOCAL) / 'images/val').glob('*.png')) +
    list((Path(DATASET_LOCAL) / 'images/val').glob('*.jpg'))
)

import random as _random
samples = _random.sample(val_images, min(4, len(val_images)))

fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 6, 5))
if len(samples) == 1:
    axes = [axes]

for ax, img_path in zip(axes, samples):
    raw  = np.fromfile(str(img_path), dtype=np.uint8)
    img  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    results = best_model(img, verbose=False, imgsz=IMG_SIZE)[0]

    # 缩放用于显示
    scale = 640 / max(w, h)
    disp  = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    dh, dw = disp.shape[:2]

    if results.masks is not None and len(results.masks) > 0:
        poly_norm = results.masks.xyn[0]
        pts = (poly_norm * [dw, dh]).astype(np.int32).reshape(-1, 1, 2)
        ov  = disp.copy()
        cv2.fillPoly(ov, [pts], (0, 255, 0))
        cv2.addWeighted(ov, 0.4, disp, 0.6, 0, disp)
        cv2.polylines(disp, [pts], True, (0, 220, 0), 2)
        conf  = float(results.boxes.conf[0])
        title = f'{img_path.name}\nconf={conf:.2f}'
    elif results.boxes is not None and len(results.boxes) > 0:
        x1, y1, x2, y2 = [int(v * scale) for v in results.boxes.xyxy[0].tolist()]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        conf  = float(results.boxes.conf[0])
        title = f'{img_path.name}\nconf={conf:.2f}'
    else:
        title = f'{img_path.name}\n(未检测到目标)'

    ax.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

# ── 推理速度测试 ─────────────────────────────────────────────────────────────
print('\n⚡ 推理速度测试（用验证集前 10 张）:')
test_imgs = val_images[:10]
times = []
for img_path in test_imgs:
    raw = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    t   = time.time()
    best_model(img, verbose=False, imgsz=IMG_SIZE)
    times.append((time.time() - t) * 1000)

avg_ms = sum(times) / len(times)
print(f'   平均推理时间 : {avg_ms:.1f} ms / 张')
if avg_ms < 200:
    print(f'   ✅ {avg_ms:.1f} ms < 200 ms，满足毕设推理速度要求！')
else:
    print(f'   ⚠️  {avg_ms:.1f} ms > 200 ms，建议换用更小的模型（yolo11n/m-seg）')


# %% [markdown]
# ---
# ## ☁️ Step 8 — 上传模型到 HuggingFace | Upload to HuggingFace


# %%
# @title ☁️ Step 8 — 上传模型到 HuggingFace | Upload to HuggingFace
# @markdown 将训练结果（模型权重 + 训练曲线 + 配置文件）上传到 HuggingFace
# @markdown
# @markdown 上传内容：
# @markdown - `weights/best.pt` — 最优模型权重 ⭐
# @markdown - `weights/last.pt` — 最终轮次权重
# @markdown - `results.png` — 训练曲线图
# @markdown - `results.csv` — 每轮指标数据
# @markdown - `args.yaml` — 训练配置记录

import time
from pathlib import Path
from huggingface_hub import CommitOperationAdd

run_path = Path(RUN_DIR)

# 决定上传到哪个仓库（默认与数据集同一个 repo 的 runs/ 目录）
UPLOAD_REPO      = DATASET_REPO   # @param {type:"string"}
UPLOAD_REPO_TYPE = 'dataset'      # @param ["dataset", "model"]

# 要上传的文件（按优先级排序）
UPLOAD_FILES = [
    ('weights/best.pt',                   '🏆 最优模型'),
    ('weights/last.pt',                   '📌 最终模型'),
    ('results.png',                       '📈 训练曲线'),
    ('results.csv',                       '📊 指标数据'),
    ('args.yaml',                         '⚙️  训练配置'),
    ('confusion_matrix_normalized.png',   '📉 混淆矩阵'),
    ('PR_curve.png',                      '📈 PR 曲线'),
    ('F1_curve.png',                      '📈 F1 曲线'),
]

SEP = '=' * 50
print(f'📤 准备上传训练结果...')
print(f'   来源目录  : {RUN_DIR}')
print(f'   目标仓库  : {UPLOAD_REPO}')
print(f'   目标路径  : runs/{RUN_NAME}/\n')

operations = []
upload_log = []

for rel_path, desc in UPLOAD_FILES:
    local_file = run_path / rel_path
    if not local_file.exists():
        print(f'   ⏭️  跳过 {rel_path}（文件不存在）')
        continue
    operations.append(CommitOperationAdd(
        path_in_repo=f'runs/{RUN_NAME}/{rel_path}',
        path_or_fileobj=local_file.read_bytes(),
    ))
    upload_log.append((desc, rel_path, local_file.stat().st_size))
    print(f'   ✅ 已准备 {desc:<12} {rel_path}  ({local_file.stat().st_size/1e6:.1f} MB)')

if not operations:
    print('\n❌ 没有找到任何可上传的文件，请确认训练已正常完成')
else:
    print(f'\n📤 正在提交 {len(operations)} 个文件到 HuggingFace...')
    t_up = time.time()

    api.create_commit(
        repo_id=UPLOAD_REPO,
        repo_type=UPLOAD_REPO_TYPE,
        operations=operations,
        commit_message=f'Training results: {RUN_NAME}',
        token=HF_TOKEN,
    )

    elapsed_up = time.time() - t_up
    print(f'\n{SEP}')
    print(f'🎉 上传完成！耗时 {elapsed_up:.1f} 秒')
    print(f'   查看地址: https://huggingface.co/datasets/{UPLOAD_REPO}/tree/main/runs/{RUN_NAME}')
    print(f'   最优权重: https://huggingface.co/datasets/{UPLOAD_REPO}/resolve/main/runs/{RUN_NAME}/weights/best.pt')
    print(f'{SEP}')


# %% [markdown]
# ---
# ## 🏁 完成 | Done
#
# 恭喜！完整训练流程结束。
#
# | 产出物 | 位置 |
# |--------|------|
# | 最优模型权重 | `runs/{RUN_NAME}/weights/best.pt` |
# | 训练曲线图 | `runs/{RUN_NAME}/results.png` |
# | 每轮指标 | `runs/{RUN_NAME}/results.csv` |
#
# ### 下载最优模型到本地
# 在新单元格执行：
# ```python
# from huggingface_hub import hf_hub_download
# path = hf_hub_download(
#     repo_id='rick003/11111',
#     filename=f'runs/{RUN_NAME}/weights/best.pt',
#     repo_type='dataset',
#     token=HF_TOKEN,
# )
# print(f'已下载到: {path}')
# ```
#
# ### 在本地推理
# ```python
# from ultralytics import YOLO
# model = YOLO('best.pt')
# results = model('your_image.jpg')
# results[0].show()
# ```


# %%
# @title 🏁 [可选] 下载最优权重到本地 Colab 文件系统 | Download Best Weights
# @markdown 运行后可在左侧文件面板右键下载 best.pt 到本机

from huggingface_hub import hf_hub_download
from google.colab import files

print(f'📥 从 HuggingFace 下载最优权重...')
local_best = hf_hub_download(
    repo_id=UPLOAD_REPO,
    filename=f'runs/{RUN_NAME}/weights/best.pt',
    repo_type=UPLOAD_REPO_TYPE,
    token=HF_TOKEN,
)
print(f'✅ 已下载到: {local_best}')

# 复制到当前工作目录，方便在文件面板找到
import shutil
dst = f'/content/best_{RUN_NAME}.pt'
shutil.copy2(local_best, dst)
print(f'📂 已复制到: {dst}')
print(f'   在左侧「文件」面板找到 best_{RUN_NAME}.pt，右键 → 下载 即可保存到本机')
