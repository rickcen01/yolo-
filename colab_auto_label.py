# ============================================================
#  气瓶图像自动标注 — Google Colab Pipeline
#  Gas Cylinder Auto-Labeling Pipeline for Google Colab
#
#  流程: 下载图像 → YOLO11x-seg 分割标注 → 上传标签到 HuggingFace
#
#  使用方法:
#  1. 打开 https://colab.research.google.com
#  2. 上传此文件：文件 → 上传笔记本 → 选择此 .py 文件
#  3. 启用 GPU：运行时 → 更改运行时类型 → T4 GPU
#  4. 从上到下依次运行每个单元格
# ============================================================


# %% [markdown]
# # 🔬 气瓶图像自动标注 — Google Colab Pipeline
#
# 本笔记本自动完成：**下载图像 → YOLO11x-seg 分割标注 → 上传标签** 到 HuggingFace
#
# **⚠️ 运行前请确认：**
# - 已启用 GPU（菜单：**运行时 → 更改运行时类型 → T4 GPU**）
# - 拥有 HuggingFace **write** 权限 Token（获取地址：https://huggingface.co/settings/tokens）
#
# **流程：**
# 1. ⚙️ 安装依赖
# 2. 🌐 [可选] cpolar 远程隧道
# 3. 🔑 HuggingFace 认证
# 4. 📊 检查标注进度
# 5. 🤖 加载 YOLO 标注引擎
# 6. 🚀 批量标注并上传
# 7. ✅ 验证结果


# %% [markdown]
# ---
# ## ⚙️ Step 1 — 安装依赖 | Install Dependencies


# %%
# @title ⚙️ Step 1 — 安装依赖 | Install Dependencies
# @markdown 首次运行时安装所需依赖，后续运行可跳过

get_ipython().system('pip install ultralytics huggingface_hub tqdm requests -q')

import torch

print('✅ 依赖安装完成')
print(f'   PyTorch 版本 : {torch.__version__}')
print(f'   CUDA 可用    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU 型号     : {torch.cuda.get_device_name(0)}')
else:
    print('   ⚠️  未检测到 GPU，建议在「运行时 → 更改运行时类型」中选择 T4 GPU')


# %% [markdown]
# ---
# ## 🌐 Step 2 — [可选] cpolar 远程隧道 | Optional Remote Tunnel


# %%
# @title 🌐 [可选] cpolar 远程隧道 | Optional Remote Access
# @markdown 如需通过手机/外部浏览器远程监控 Colab 运行状态，请：
# @markdown 1. 在 https://dashboard.cpolar.com 注册并获取 authtoken
# @markdown 2. 将 token 填入下方，并勾选 ENABLE_CPOLAR
# @markdown
# @markdown 如不需要远程监控，保持 ENABLE_CPOLAR = False 直接跳过即可

ENABLE_CPOLAR = False  # @param {type:"boolean"}
CPOLAR_TOKEN  = ''     # @param {type:"string"}

if ENABLE_CPOLAR:
    if not CPOLAR_TOKEN:
        print('❌ 请填写 CPOLAR_TOKEN 后再勾选启用')
    else:
        import os, time, re

        print('📦 正在安装 cpolar...')
        os.system(
            'curl -sL https://www.cpolar.com/static/downloads/'
            'install-release-cpolar.sh | sudo bash'
        )

        print('🔑 正在认证 cpolar...')
        ret = os.system(f'cpolar authtoken {CPOLAR_TOKEN}')
        if ret != 0:
            print('❌ authtoken 设置失败，请检查 Token 是否正确')
        else:
            print('🚀 正在启动隧道...')
            os.system('nohup cpolar http 8888 --log-level=warn > /tmp/cpolar.log 2>&1 &')
            time.sleep(8)

            try:
                log  = open('/tmp/cpolar.log').read()
                urls = re.findall(r'https://\S+\.cpolar\.cn', log)
                if urls:
                    print('✅ cpolar 隧道已启动！')
                    print(f'🔗 外部访问地址: {urls[0]}')
                    print('   （在外部浏览器中打开此链接即可远程查看 Colab）')
                else:
                    print('⏳ 隧道启动中，稍后执行以下命令查看地址:')
                    print('   !cat /tmp/cpolar.log | grep cpolar.cn')
            except Exception as e:
                print(f'⚠️  日志读取失败: {e}')
                print('   请手动执行: !cat /tmp/cpolar.log')
else:
    print('ℹ️  已跳过 cpolar 设置')
    print('   如需远程监控，将 ENABLE_CPOLAR 改为 True 并填写 CPOLAR_TOKEN')


# %% [markdown]
# ---
# ## 🔑 Step 3 — HuggingFace 认证 | Authentication


# %%
# @title 🔑 Step 3 — HuggingFace 认证 | Authentication
# @markdown 填入 HuggingFace Token（需要 **write** 权限）
# @markdown
# @markdown 获取地址：https://huggingface.co/settings/tokens → New token → Role: Write

HF_TOKEN     = ''              # @param {type:"string"}
DATASET_REPO = 'rick003/11111' # @param {type:"string"}

import os
from huggingface_hub import login, HfApi

if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('请输入 HuggingFace Token（输入内容不可见）: ')

if not HF_TOKEN:
    raise ValueError('❌ 必须提供 HuggingFace Token 才能继续')

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

# 验证数据集是否可访问
try:
    info = api.dataset_info(DATASET_REPO)
    print(f'✅ HuggingFace 认证成功')
    print(f'📦 数据集     : {DATASET_REPO}')
    print(f'🔗 访问地址   : https://huggingface.co/datasets/{DATASET_REPO}')
except Exception as e:
    print(f'❌ 数据集访问失败: {e}')
    print('   请检查 Token 是否有 write 权限，以及数据集名称是否正确')


# %% [markdown]
# ---
# ## 📊 Step 4 — 检查标注进度 | Check Labeling Progress


# %%
# @title 📊 Step 4 — 检查标注进度 | Check Labeling Progress
# @markdown 扫描 HuggingFace 数据集，找出哪些图像尚未标注
# @markdown
# @markdown **REGENERATE_ALL**: 勾选后将对所有图像重新生成标注（含已有标注的图像）

REGENERATE_ALL = False  # @param {type:"boolean"}

print('🔍 正在扫描 HuggingFace 数据集文件列表...')
print('   （文件较多时可能需要 10-30 秒）')

all_files = list(api.list_repo_files(DATASET_REPO, repo_type='dataset'))

image_files = sorted([
    f for f in all_files
    if f.startswith('images/train/')
    and f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
label_files = sorted([
    f for f in all_files
    if f.startswith('labels/train/') and f.endswith('.txt')
])

labeled_names = set(
    os.path.splitext(os.path.basename(f))[0] for f in label_files
)
image_info  = [
    (f, os.path.splitext(os.path.basename(f))[0]) for f in image_files
]
done_images = [(f, n) for f, n in image_info if n in labeled_names]
unlabeled   = [(f, n) for f, n in image_info if n not in labeled_names]

if REGENERATE_ALL:
    todo_images = image_info
else:
    todo_images = unlabeled

SEP = '=' * 50
print(f'\n{SEP}')
print(f'📊 扫描结果:')
print(f'   图片总数 : {len(image_info):>5} 张')
print(f'   已有标注 : {len(done_images):>5} 张')
print(f'   待处理   : {len(todo_images):>5} 张')
print(f'{SEP}')

if REGENERATE_ALL:
    print('⚠️  模式: 重新生成全部标注（REGENERATE_ALL = True）')

if not todo_images:
    print('\n🎉 所有图片已标注完毕！')
    print('   如需重新生成标注，请勾选上方的 REGENERATE_ALL 选项')
else:
    print(f'\n➡️  将处理 {len(todo_images)} 张图像，请继续运行下方单元格')


# %% [markdown]
# ---
# ## 🤖 Step 5 — 加载标注引擎 | Load Annotation Engine


# %%
# @title 🤖 Step 5 — 加载标注引擎 | Load Annotation Engine
# @markdown 加载 YOLO11x-seg 模型作为自动标注引擎
# @markdown
# @markdown 模型首次运行时会自动从 Ultralytics 下载（约 59 MB），请耐心等待

import cv2
import numpy as np
import torch


class GasCylinderSegmenter:
    """
    基于 YOLO11x-seg 的气瓶自动分割标注引擎
    改编自 process_E_to_F.py / auto_label_final_robust.py

    算法说明:
    - 使用极低置信度阈值 (0.01) 检测所有可能目标
    - 从 COCO 类别中筛选外形与气瓶相似的类别（bottle/fire hydrant/vase 等）
    - 通过面积比和中心距两个几何条件过滤噪声
    - 对通过筛选的候选框按综合得分排序，取最优结果
    - 输出标准 YOLO 分割格式标注字符串
    """

    # COCO 类别权重：外形与气瓶越相似权重越高
    CLASS_WEIGHTS = {
        39: 2.0,   # bottle     — 最理想匹配
        10: 1.0,   # fire hydrant
        71: 1.0,   # sink
        61: 1.0,   # toilet
        41: 0.8,   # cup
        56: 0.8,   # chair
        75: 0.8,   # vase
    }

    def __init__(self, model_path='yolo11x-seg.pt'):
        from ultralytics import YOLO

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        dev_name = (
            torch.cuda.get_device_name(0) if self.device == '0' else 'CPU (慢)'
        )
        print(f'🎮 推理设备  : {dev_name}')
        print(f'📥 加载模型  : {model_path}')
        print('   首次运行会自动下载模型文件，请稍候...')

        self.model    = YOLO(model_path)
        self.conf     = 0.01   # 极低阈值，依赖几何过滤代替置信度过滤
        self.min_area = 0.03   # 目标面积至少占图像 3%
        self.max_dist = 0.35   # 目标中心距图像中心不超过对角线的 35%

        print('✅ 模型加载完成')

    def annotate(self, img):
        """
        对单张图像进行推理，返回 (label_str, success)

        label_str : YOLO 分割格式字符串，例如:
                    "0 0.123456 0.234567 0.345678 ..."
                    （0 = gas_cylinder 类别 ID，后续为归一化多边形坐标）
        success   : True 表示检测到有效目标，False 表示未检测到
        """
        h, w   = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        area   = float(w * h)
        diag   = float(w**2 + h**2) ** 0.5

        res = self.model(img, conf=self.conf, device=self.device, verbose=False)[0]

        candidates = []
        if res.masks is not None:
            for i in range(len(res.masks.data)):
                cid  = int(res.boxes.cls[i])
                conf = float(res.boxes.conf[i])

                if cid not in self.CLASS_WEIGHTS:
                    continue

                x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
                box_cx = (x1 + x2) / 2.0
                box_cy = (y1 + y2) / 2.0

                ar = (x2 - x1) * (y2 - y1) / area
                dr = ((box_cx - cx)**2 + (box_cy - cy)**2) ** 0.5 / diag

                # 几何过滤
                if ar < self.min_area or dr > self.max_dist:
                    continue

                # 综合得分 = 置信度 × 类别权重 × 中心度加成
                centrality_bonus = 1.0 + (1.0 - dr / self.max_dist)
                score = conf * self.CLASS_WEIGHTS[cid] * centrality_bonus

                candidates.append({
                    'score': score,
                    'conf' : conf,
                    'name' : res.names[cid],
                    'poly' : res.masks.xyn[i],
                })

        if not candidates:
            return '', False

        best = max(candidates, key=lambda x: x['score'])
        poly = best['poly']

        if len(poly) < 3:
            return '', False

        # 构建 YOLO 分割标注字符串
        pts = ['0']
        for p in poly:
            pts.append(f'{float(p[0]):.6f}')
            pts.append(f'{float(p[1]):.6f}')

        label_str = ' '.join(pts)
        return label_str, True


# 初始化标注引擎
segmenter = GasCylinderSegmenter('yolo11x-seg.pt')
print('\n✅ 标注引擎就绪，可以开始处理图像！')


# %% [markdown]
# ---
# ## 🚀 Step 6 — 批量标注并上传 | Batch Annotate & Upload


# %%
# @title 🚀 Step 6 — 批量标注并上传 | Batch Annotate & Upload
# @markdown **BATCH_SIZE**: 每批处理的图像数量（建议 5，内存不足时调小）
# @markdown
# @markdown **SAVE_VISUALIZATIONS**: 是否将带标注叠加的预览图上传到 HF `preview/train/` 目录
# @markdown （开启后便于质检，但会增加上传时间）

BATCH_SIZE          = 5      # @param {type:"integer"}
SAVE_VISUALIZATIONS = False  # @param {type:"boolean"}

import requests
import time
from pathlib import Path
from huggingface_hub import CommitOperationAdd
from tqdm.notebook import tqdm

# 工作目录（Colab 临时存储，运行期间有效）
WORK = Path('/tmp/gas_anno')
IMGS = WORK / 'images'
VISF = WORK / 'visual'
for d in [IMGS, VISF]:
    d.mkdir(parents=True, exist_ok=True)

# 统计信息
stats  = {'success': 0, 'failed': 0, 'error': 0, 'uploaded': 0}
missed = []   # 未检测到目标的图像列表
t0     = time.time()


# ── 工具函数 ────────────────────────────────────────────────────────────────

def download_image(hf_path, dest_path):
    """从 HuggingFace 下载单张图像"""
    url = (
        f'https://huggingface.co/datasets/{DATASET_REPO}'
        f'/resolve/main/{hf_path}'
    )
    try:
        r = requests.get(
            url,
            headers={'Authorization': f'Bearer {HF_TOKEN}'},
            stream=True,
            timeout=180,
        )
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 17):  # 128 KB chunks
                f.write(chunk)
        return True
    except Exception as e:
        tqdm.write(f'    ❌ 下载失败 [{Path(hf_path).name}]: {e}')
        return False


def draw_visualization(img, label_str, name):
    """在图像上绘制分割多边形叠加层，保存到 VISF 目录"""
    try:
        h, w = img.shape[:2]
        vals = [float(v) for v in label_str.split()[1:]]
        arr  = (
            np.array(vals).reshape(-1, 2) * [w, h]
        ).astype(np.int32).reshape(-1, 1, 2)

        overlay = img.copy()
        cv2.fillPoly(overlay, [arr], (0, 255, 0))
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.polylines(img, [arr], True, (0, 255, 0), 2)

        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            (VISF / f'{name}.jpg').write_bytes(buf.tobytes())
    except Exception:
        pass  # 可视化失败不影响主流程


def commit_batch_labels(label_map, vis_map):
    """
    将一批标签文件以单次 commit 提交到 HuggingFace。
    单次 commit 比逐文件上传快得多，且减少 Git 历史污染。
    """
    operations = []

    # 标签文件
    for name, label_str in label_map.items():
        operations.append(CommitOperationAdd(
            path_in_repo=f'labels/train/{name}.txt',
            path_or_fileobj=label_str.encode('utf-8'),
        ))

    # 可视化预览图（可选）
    if SAVE_VISUALIZATIONS:
        for name, vis_path in vis_map.items():
            vp = Path(vis_path)
            if vp.exists():
                operations.append(CommitOperationAdd(
                    path_in_repo=f'preview/train/{name}.jpg',
                    path_or_fileobj=vp.read_bytes(),
                ))

    if not operations:
        return 0

    keys = list(label_map.keys())
    commit_msg = (
        f'auto-label: {keys[0]} ~ {keys[-1]}'
        f' ({len(label_map)} files)'
    )
    api.create_commit(
        repo_id=DATASET_REPO,
        repo_type='dataset',
        operations=operations,
        commit_message=commit_msg,
    )
    return len(label_map)


# ── 主循环 ─────────────────────────────────────────────────────────────────

if not todo_images:
    print('✅ 没有需要处理的图像。')
    print('   若需重新生成标注，请在 Step 4 中勾选 REGENERATE_ALL 后重新运行。')

else:
    batches = [
        todo_images[i:i + BATCH_SIZE]
        for i in range(0, len(todo_images), BATCH_SIZE)
    ]
    total_batches = len(batches)

    print(f'📋 共 {len(todo_images)} 张待处理，'
          f'分 {total_batches} 批（每批最多 {BATCH_SIZE} 张）')
    print(f'💾 工作目录: {WORK}')
    print(f'🔄 开始处理...\n')

    for bi, batch in enumerate(tqdm(batches, desc='总体进度')):
        batch_names = [n for _, n in batch]
        tqdm.write(f'\n{"─"*55}')
        tqdm.write(f'批次 {bi+1}/{total_batches}: {batch_names}')
        tqdm.write(f'{"─"*55}')

        label_map = {}
        vis_map   = {}

        # ── 阶段 1: 下载 ──────────────────────────────────────────────────
        tqdm.write('  [1/3] 下载图像...')
        for hfp, name in batch:
            dest = IMGS / Path(hfp).name
            if download_image(hfp, dest):
                tqdm.write(f'    ⬇️  {name} ({dest.stat().st_size / 1e6:.1f} MB)')
            else:
                stats['error'] += 1

        # ── 阶段 2: 标注 ──────────────────────────────────────────────────
        tqdm.write('  [2/3] 执行 YOLO 推理标注...')
        for hfp, name in batch:
            src = IMGS / Path(hfp).name
            if not src.exists():
                tqdm.write(f'    ⚠️  跳过 {name}（下载失败）')
                continue

            try:
                # 使用 numpy 读取避免中文路径问题
                raw = np.fromfile(str(src), dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError('cv2.imdecode 返回 None，图像可能已损坏')

                label_str, ok = segmenter.annotate(img)

                if ok:
                    label_map[name] = label_str
                    stats['success'] += 1
                    n_pts = (len(label_str.split()) - 1) // 2
                    tqdm.write(f'    ✅ {name}  → {n_pts} 个多边形点')

                    if SAVE_VISUALIZATIONS:
                        draw_visualization(img.copy(), label_str, name)
                        vis_map[name] = str(VISF / f'{name}.jpg')
                else:
                    # 未检测到目标：写入空标注文件（标记为已处理）
                    label_map[name] = ''
                    stats['failed'] += 1
                    missed.append(name)
                    tqdm.write(f'    ⚠️  {name}  → 未检测到目标（写入空标注）')

            except Exception as e:
                stats['error'] += 1
                tqdm.write(f'    ❌ {name}  → 处理出错: {e}')

        # ── 阶段 3: 上传 ──────────────────────────────────────────────────
        if label_map:
            tqdm.write(f'  [3/3] 上传 {len(label_map)} 个标签到 HuggingFace...')
            try:
                n_uploaded = commit_batch_labels(label_map, vis_map)
                stats['uploaded'] += n_uploaded
                tqdm.write(f'    📤 成功上传 {n_uploaded} 个文件')
            except Exception as e:
                tqdm.write(f'    ❌ 上传失败: {e}')
                tqdm.write(f'       标签已保存在本地: {WORK}/labels/')
        else:
            tqdm.write('  [3/3] 本批次无标签需要上传')

        # ── 清理磁盘（节省 Colab 存储空间）─────────────────────────────────
        for hfp, _ in batch:
            img_file = IMGS / Path(hfp).name
            img_file.unlink(missing_ok=True)
        for vf in VISF.iterdir():
            vf.unlink(missing_ok=True)

        # ── 进度与 ETA ───────────────────────────────────────────────────
        elapsed     = time.time() - t0
        imgs_done   = min((bi + 1) * BATCH_SIZE, len(todo_images))
        imgs_remain = max(len(todo_images) - imgs_done, 0)
        eta_sec     = (elapsed / max(imgs_done, 1)) * imgs_remain

        s_ok = stats['success']
        s_fa = stats['failed']
        s_er = stats['error']
        s_up = stats['uploaded']

        tqdm.write(
            f'  ⏱️  已用 {elapsed/60:.1f} 分 | 剩余≈{eta_sec/60:.1f} 分  |  '
            f'✅{s_ok}  ⚠️{s_fa}  ❌{s_er}  ⬆️{s_up}'
        )

    # ── 最终汇总 ──────────────────────────────────────────────────────────
    total_time = time.time() - t0
    SEP = '=' * 55

    print(f'\n{SEP}')
    print('🎉 批量标注完成！')
    print(f'{SEP}')
    print(f'   ✅ 成功标注   : {stats["success"]:>5} 张')
    print(f'   ⚠️  未检测到   : {stats["failed"]:>5} 张（已写入空标注）')
    print(f'   ❌ 处理错误   : {stats["error"]:>5} 张')
    print(f'   📤 已上传标签 : {stats["uploaded"]:>5} 个')
    print(f'   ⏱️  总耗时     :  {total_time/60:.1f} 分钟')
    print(f'{SEP}')

    if missed:
        print(f'\n⚠️  以下 {len(missed)} 张图像未检测到气瓶，建议人工复查并补充标注:')
        for n in missed[:30]:
            print(f'     - {n}')
        if len(missed) > 30:
            print(f'     ... 共 {len(missed)} 张，完整列表见变量 `missed`')


# %% [markdown]
# ---
# ## ✅ Step 7 — 验证结果 | Verify Results


# %%
# @title ✅ Step 7 — 验证结果 | Verify Upload Results
# @markdown 重新扫描 HuggingFace 数据集，确认标注文件已正确上传

print('🔍 正在验证 HuggingFace 最新状态...\n')

files_now = list(api.list_repo_files(DATASET_REPO, repo_type='dataset'))

img_count = len([
    f for f in files_now if f.startswith('images/train/')
])
lbl_count = len([
    f for f in files_now
    if f.startswith('labels/train/') and f.endswith('.txt')
])
pre_count = len([
    f for f in files_now
    if f.startswith('preview/train/') and f.endswith('.jpg')
])
coverage = lbl_count / img_count * 100 if img_count else 0.0

SEP = '=' * 55
print(f'{SEP}')
print('📊 数据集最新状态:')
print(f'{SEP}')
print(f'   🖼️  图片总数    : {img_count:>5} 张')
print(f'   🏷️  标签总数    : {lbl_count:>5} 个')
print(f'   🖼️  预览图总数  : {pre_count:>5} 张')
print(f'   📈 标注覆盖率  : {coverage:>5.1f} %')
print(f'{SEP}')
print(f'\n🔗 数据集地址: https://huggingface.co/datasets/{DATASET_REPO}')

if coverage < 100.0:
    still_missing = img_count - lbl_count
    print(f'\n⚠️  仍有 {still_missing} 张图像缺少标注。')
    print('   可重新运行 Step 4 → Step 6 继续处理剩余图像。')
else:
    print('\n🎉 所有图像均已标注，覆盖率 100%！')


# %% [markdown]
# ---
# ## 🧹 Step 8 — 数据清洗建议 | Data Cleaning Guide
#
# 完成自动标注后，建议在训练前对标注结果做以下清洗：
#
# | 步骤 | 说明 |
# |------|------|
# | 1. 人工抽检 | 打开 `preview/train/` 目录，随机抽查 20~30 张预览图，确认分割轮廓贴合气瓶 |
# | 2. 清理空标注 | 上方 `missed` 列表中的图像标注为空，需手动补充或直接从数据集删除 |
# | 3. 检查异常形状 | 多边形点数过少（< 10 点）或过多（> 200 点）的标注可能质量差，建议复查 |
# | 4. 去除重复图像 | 使用感知哈希（pHash）检测相似度 > 95% 的重复图像并删除 |
# | 5. 验证格式 | 确认所有 `.txt` 文件符合 YOLO 分割格式：`class_id x1 y1 x2 y2 ...` |


# %%
# @title 🧹 Step 8 — 数据清洗 | Data Cleaning
# @markdown 对自动标注结果进行质量检查和清洗
# @markdown
# @markdown **CHECK_EMPTY_LABELS**: 列出所有空标注文件（未检测到目标）
# @markdown
# @markdown **CHECK_POINT_COUNT**: 检查多边形点数异常的标注

CHECK_EMPTY_LABELS = True   # @param {type:"boolean"}
CHECK_POINT_COUNT  = True   # @param {type:"boolean"}
MIN_POINTS         = 8      # @param {type:"integer"}
MAX_POINTS         = 200    # @param {type:"integer"}

print('🧹 开始数据清洗检查...\n')

# 重新获取最新文件列表
all_files_clean = list(api.list_repo_files(DATASET_REPO, repo_type='dataset'))
label_files_clean = [
    f for f in all_files_clean
    if f.startswith('labels/train/') and f.endswith('.txt')
]

import requests

empty_labels    = []
few_pts_labels  = []
many_pts_labels = []
checked         = 0
errors          = 0

print(f'📋 共扫描 {len(label_files_clean)} 个标签文件...')
print('   （每 50 个文件汇报一次进度）\n')

for i, lbl_path in enumerate(label_files_clean):
    name = os.path.splitext(os.path.basename(lbl_path))[0]

    try:
        url = (
            f'https://huggingface.co/datasets/{DATASET_REPO}'
            f'/resolve/main/{lbl_path}'
        )
        r = requests.get(
            url,
            headers={'Authorization': f'Bearer {HF_TOKEN}'},
            timeout=30,
        )
        r.raise_for_status()
        content = r.text.strip()
        checked += 1

        if CHECK_EMPTY_LABELS and not content:
            empty_labels.append(name)
            continue

        if CHECK_POINT_COUNT and content:
            parts  = content.split()
            n_pts  = (len(parts) - 1) // 2   # 去掉 class_id 后剩余坐标对数
            if n_pts < MIN_POINTS:
                few_pts_labels.append((name, n_pts))
            elif n_pts > MAX_POINTS:
                many_pts_labels.append((name, n_pts))

    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f'   ⚠️  读取失败 {name}: {e}')

    if (i + 1) % 50 == 0:
        print(f'   已检查 {i+1}/{len(label_files_clean)} ...')

# ── 汇报结果 ─────────────────────────────────────────────────────────────
SEP = '=' * 55
print(f'\n{SEP}')
print('🧹 数据清洗检查结果:')
print(f'{SEP}')
print(f'   已检查标签   : {checked} 个')
print(f'   读取错误     : {errors} 个')
print(f'   空标注文件   : {len(empty_labels)} 个  ← 未检测到气瓶，需人工处理')
print(f'   点数过少(<{MIN_POINTS}): {len(few_pts_labels)} 个  ← 轮廓粗糙，建议复查')
print(f'   点数过多(>{MAX_POINTS}): {len(many_pts_labels)} 个  ← 可能过拟合噪声，建议复查')
print(f'{SEP}')

if empty_labels:
    print(f'\n📋 空标注文件列表（前 20 条）:')
    for n in empty_labels[:20]:
        print(f'   - {n}')
    if len(empty_labels) > 20:
        print(f'   ... 共 {len(empty_labels)} 个，完整列表见变量 `empty_labels`')

if few_pts_labels:
    print(f'\n📋 点数过少的标签（前 10 条）:')
    for n, pts in few_pts_labels[:10]:
        print(f'   - {n}  ({pts} 点)')

if many_pts_labels:
    print(f'\n📋 点数过多的标签（前 10 条）:')
    for n, pts in many_pts_labels[:10]:
        print(f'   - {n}  ({pts} 点)')

print(f'\n{SEP}')
print('📌 后续建议:')
print('   1. 对 empty_labels 中的图像使用 LabelImg / Label Studio 手动标注')
print('   2. 对 few_pts_labels 中的标注在可视化预览中人工核对')
print('   3. 清洗完成后即可在本地或 Colab 中开始 YOLO 训练:')
print('      from ultralytics import YOLO')
print('      model = YOLO("yolo11x-seg.pt")')
print('      model.train(data="dataset.yaml", epochs=100, imgsz=640)')
print(f'{SEP}')
