# ============================================================
#  数据集检测与清洗 — Dataset Inspection & Cleaning
#  Google Colab Script
#
#  检测内容:
#  1. 空标签文件（0 字节）
#  2. 类别 ID 错误（非 0）
#  3. 坐标越界（不在 0~1 范围）
#  4. 多边形点数不足（< 3 个点）
#  5. 坐标数量奇偶错误
#  6. 图像缺少对应标签
#  7. 标签缺少对应图像
#
#  修复内容:
#  1. 自动将类别 ID 改为 0
#  2. 自动将越界坐标裁剪到 [0, 1]
#  3. 删除空标签文件（YOLO 训练时缺失标签等同于空标签，但 copy_paste 增强会崩溃）
#  4. 删除点数不足 / 格式错误的标注行
#  5. 删除旧的 .cache 文件和 .cache 目录
#  6. 将修复后的标签同步回 HuggingFace
# ============================================================


# %% [markdown]
# # 🧹 数据集检测与清洗
#
# 本脚本对 HuggingFace 数据集的 train / val 标签进行全面检测，
# 自动修复能修复的问题，并将结果同步回 HuggingFace。


# %% [markdown]
# ---
# ## ⚙️ Step 1 — 安装依赖与认证 | Setup


# %%
# @title ⚙️ Step 1 — 安装依赖与认证 | Setup
# @markdown 填入 HuggingFace Token（需要 write 权限）

HF_TOKEN     = ''              # @param {type:"string"}
DATASET_REPO = 'rick003/11111' # @param {type:"string"}
DATASET_LOCAL = '/content/gas_dataset'

get_ipython().system('pip install huggingface_hub -q')

import os, shutil
from pathlib import Path
from huggingface_hub import login, HfApi, snapshot_download

if not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass('请输入 HuggingFace Token: ')

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()
print('✅ 认证成功')
print(f'   数据集: {DATASET_REPO}')


# %% [markdown]
# ---
# ## 📥 Step 2 — 下载数据集（如已下载可跳过）| Download Dataset


# %%
# @title 📥 Step 2 — 下载数据集（如已下载可跳过）| Download Dataset
# @markdown 如果 `/content/gas_dataset` 已存在且完整，可直接跳到 Step 3

FORCE_REDOWNLOAD = False  # @param {type:"boolean"}

base = Path(DATASET_LOCAL)

if base.exists() and not FORCE_REDOWNLOAD:
    train_imgs = list((base / 'images/train').glob('*.png')) + \
                 list((base / 'images/train').glob('*.jpg'))
    val_imgs   = list((base / 'images/val').glob('*.png')) + \
                 list((base / 'images/val').glob('*.jpg'))
    if len(train_imgs) > 0:
        print(f'✅ 数据集已存在，跳过下载')
        print(f'   train 图像: {len(train_imgs)} 张')
        print(f'   val   图像: {len(val_imgs)} 张')
    else:
        FORCE_REDOWNLOAD = True
        print('⚠️  目录存在但为空，重新下载')

if FORCE_REDOWNLOAD:
    print(f'📥 下载数据集到 {DATASET_LOCAL} ...')
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type='dataset',
        token=HF_TOKEN,
        local_dir=DATASET_LOCAL,
        ignore_patterns=['preview/**', 'runs/**', '*.cache', '.gitattributes'],
    )
    print('✅ 下载完成')


# %% [markdown]
# ---
# ## 🔍 Step 3 — 全面检测数据集 | Full Inspection


# %%
# @title 🔍 Step 3 — 全面检测数据集 | Full Inspection

import numpy as np

base = Path(DATASET_LOCAL)
SEP  = '=' * 60

# ── 问题容器 ─────────────────────────────────────────────────────────────────
issues = {
    'empty_label'    : [],   # (split, name)        — 空标签文件
    'wrong_class'    : [],   # (split, name, cls_ids) — 类别 ID 非 0
    'coord_oob'      : [],   # (split, name)        — 坐标越界
    'bad_polygon'    : [],   # (split, name)        — 点数不足或坐标数奇偶错
    'no_label'       : [],   # (split, name)        — 图像缺对应标签
    'no_image'       : [],   # (split, name)        — 标签缺对应图像
}

IMG_EXTS = {'.png', '.jpg', '.jpeg'}

def inspect_label_file(lbl_path):
    """
    检测单个标签文件。
    返回 dict: {
        'empty'      : bool,
        'wrong_class': [int, ...],   非 0 的类别 ID 列表
        'coord_oob'  : bool,         是否有越界坐标
        'bad_polygon': bool,         是否有格式错误的行
    }
    """
    result = dict(empty=False, wrong_class=[], coord_oob=False, bad_polygon=False)

    text = lbl_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        result['empty'] = True
        return result

    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        # 类别 ID
        try:
            cls_id = int(parts[0])
        except ValueError:
            result['bad_polygon'] = True
            continue

        if cls_id != 0:
            result['wrong_class'].append(cls_id)

        # 坐标部分
        coords = parts[1:]

        # 坐标数量必须是偶数且 >= 6（至少 3 个点）
        if len(coords) < 6 or len(coords) % 2 != 0:
            result['bad_polygon'] = True
            continue

        # 坐标值必须在 [0, 1]
        try:
            vals = [float(v) for v in coords]
        except ValueError:
            result['bad_polygon'] = True
            continue

        if any(v < -0.01 or v > 1.01 for v in vals):
            result['coord_oob'] = True

    return result


# ── 逐 split 扫描 ─────────────────────────────────────────────────────────────
total_labels   = 0
total_images   = 0

for split in ('train', 'val'):
    img_dir = base / f'images/{split}'
    lbl_dir = base / f'labels/{split}'

    if not img_dir.exists():
        print(f'⚠️  images/{split}/ 不存在，跳过')
        continue

    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in lbl_dir.glob('*.txt')} if lbl_dir.exists() else set()

    total_images += len(img_stems)
    total_labels += len(lbl_stems)

    print(f'\n🔍 检测 [{split}]  图像: {len(img_stems)}  标签: {len(lbl_stems)}')

    # 无标签图像
    for s in sorted(img_stems - lbl_stems):
        issues['no_label'].append((split, s))

    # 无图像标签
    for s in sorted(lbl_stems - img_stems):
        issues['no_image'].append((split, s))

    # 逐文件内容检测
    for lbl_path in sorted(lbl_dir.glob('*.txt')):
        name = lbl_path.stem
        res  = inspect_label_file(lbl_path)

        if res['empty']:
            issues['empty_label'].append((split, name))
        if res['wrong_class']:
            issues['wrong_class'].append((split, name, res['wrong_class']))
        if res['coord_oob']:
            issues['coord_oob'].append((split, name))
        if res['bad_polygon']:
            issues['bad_polygon'].append((split, name))

    # 小计
    counts = {
        '空标签'      : sum(1 for s,n   in issues['empty_label'] if s == split),
        '类别ID错误'  : sum(1 for s,n,_ in issues['wrong_class'] if s == split),
        '坐标越界'    : sum(1 for s,n   in issues['coord_oob']   if s == split),
        '格式错误'    : sum(1 for s,n   in issues['bad_polygon'] if s == split),
        '缺标签'      : sum(1 for s,n   in issues['no_label']    if s == split),
        '缺图像'      : sum(1 for s,n   in issues['no_image']    if s == split),
    }
    for k, v in counts.items():
        status = '❌' if v > 0 else '✅'
        print(f'   {status} {k:<10} : {v}')


# ── 汇总 ─────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('📊 全部问题汇总:')
print(f'{SEP}')
print(f'   总图像数          : {total_images}')
print(f'   总标签数          : {total_labels}')
print(f'   空标签文件        : {len(issues["empty_label"]):>4}  ← copy_paste 增强会崩溃')
print(f'   类别 ID 非 0      : {len(issues["wrong_class"]):>4}  ← 训练会报 class count 错误')
print(f'   坐标越界          : {len(issues["coord_oob"]):>4}  ← 标注框超出图像边界')
print(f'   多边形格式错误    : {len(issues["bad_polygon"]):>4}  ← 点数不足或坐标数奇偶错')
print(f'   图像缺对应标签    : {len(issues["no_label"]):>4}  ← 这些图像训练时无监督信号')
print(f'   标签缺对应图像    : {len(issues["no_image"]):>4}  ← 孤立标签文件')
print(f'{SEP}')

total_fixable = (len(issues['empty_label']) + len(issues['wrong_class']) +
                 len(issues['coord_oob'])   + len(issues['bad_polygon']))
total_report  = len(issues['no_label']) + len(issues['no_image'])

if total_fixable == 0 and total_report == 0:
    print('\n🎉 数据集完全干净，无需清洗！直接运行训练脚本即可。')
else:
    print(f'\n✅ 可自动修复: {total_fixable} 项')
    print(f'ℹ️  仅报告（需人工处理）: {total_report} 项')
    print('\n➡️  继续运行 Step 4 开始自动修复')


# %% [markdown]
# ---
# ## 🔧 Step 4 — 自动修复 | Auto Fix


# %%
# @title 🔧 Step 4 — 自动修复 | Auto Fix
# @markdown 自动修复可修复的问题，并删除旧缓存

MIN_POLYGON_PTS = 3  # @param {type:"integer"}
# @markdown **MIN_POLYGON_PTS**: 多边形最少点数，低于此值的标注行直接删除

fixed_files  = {}   # (split, name) -> new_content  (空字符串 = 需要删除文件)
fix_log      = []

def fix_label_content(lbl_path):
    """
    修复单个标签文件内容。
    返回 (需要修改, 新内容字符串 or None)
    新内容为 None 表示此文件应被删除（空标签 or 修复后为空）
    """
    text = lbl_path.read_text(encoding='utf-8', errors='ignore').strip()

    # 空文件 → 删除
    if not text:
        return True, None

    good_lines = []
    file_changed = False

    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        # 解析类别 ID
        try:
            cls_id = int(parts[0])
        except ValueError:
            file_changed = True   # 格式错误行，丢弃
            continue

        coords = parts[1:]

        # 坐标数量检查
        if len(coords) < MIN_POLYGON_PTS * 2 or len(coords) % 2 != 0:
            file_changed = True   # 点数不足，丢弃此行
            continue

        # 解析坐标
        try:
            vals = [float(v) for v in coords]
        except ValueError:
            file_changed = True
            continue

        # 裁剪越界坐标到 [0, 1]
        clipped = [max(0.0, min(1.0, v)) for v in vals]
        if clipped != vals:
            file_changed = True
            vals = clipped

        # 类别 ID 统一改为 0
        if cls_id != 0:
            cls_id       = 0
            file_changed = True

        coord_str = ' '.join(f'{v:.6f}' for v in vals)
        good_lines.append(f'0 {coord_str}')

    if not good_lines:
        # 修复后内容为空 → 删除文件
        return True, None

    new_content = '\n'.join(good_lines)

    if file_changed:
        return True, new_content
    return False, None


# ── 遍历修复 ────────────────────────────────────────────────────────────────
n_fixed    = 0
n_deleted  = 0
n_ok       = 0

for split in ('train', 'val'):
    lbl_dir = base / f'labels/{split}'
    if not lbl_dir.exists():
        continue

    lbl_files = sorted(lbl_dir.glob('*.txt'))
    print(f'\n🔧 修复 [{split}]: {len(lbl_files)} 个文件...')

    for lbl_path in lbl_files:
        changed, new_content = fix_label_content(lbl_path)

        if not changed:
            n_ok += 1
            continue

        if new_content is None:
            # 删除空标签文件
            lbl_path.unlink()
            fix_log.append(f'[DELETE] {split}/{lbl_path.name}')
            fixed_files[(split, lbl_path.stem)] = None
            n_deleted += 1
        else:
            # 写入修复后内容
            lbl_path.write_text(new_content, encoding='utf-8')
            fix_log.append(f'[FIX]    {split}/{lbl_path.name}')
            fixed_files[(split, lbl_path.stem)] = new_content
            n_fixed += 1

    print(f'   ✅ 无需修改: {n_ok}')
    print(f'   🔧 已修复  : {n_fixed}')
    print(f'   🗑️  已删除  : {n_deleted}（空标签）')


# ── 删除所有 .cache 文件/目录 ─────────────────────────────────────────────────
print('\n🗑️  清理旧缓存...')
removed_caches = []

for item in base.rglob('*.cache'):
    try:
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
        removed_caches.append(str(item))
    except Exception as e:
        print(f'   ⚠️  清理失败 {item}: {e}')

for name in ('train.cache', 'val.cache'):
    for p in base.rglob(name):
        try:
            if p.is_file():
                p.unlink()
                removed_caches.append(str(p))
        except Exception:
            pass

if removed_caches:
    for r in removed_caches:
        print(f'   🗑️  {r}')
else:
    print('   ℹ️  无缓存文件需要清理')


# ── 汇总 ────────────────────────────────────────────────────────────────────
SEP = '=' * 60
print(f'\n{SEP}')
print('✅ 本地修复完成:')
print(f'   修复内容的标签  : {n_fixed} 个')
print(f'   删除空标签      : {n_deleted} 个')
print(f'   清理缓存        : {len(removed_caches)} 个')
print(f'{SEP}')

if fix_log:
    print(f'\n📋 修改记录（前 30 条）:')
    for entry in fix_log[:30]:
        print(f'   {entry}')
    if len(fix_log) > 30:
        print(f'   ... 共 {len(fix_log)} 条，完整记录见变量 fix_log')

print('\n➡️  继续运行 Step 5 将修复结果同步到 HuggingFace')


# %% [markdown]
# ---
# ## ☁️ Step 5 — 同步到 HuggingFace | Sync to HuggingFace


# %%
# @title ☁️ Step 5 — 同步到 HuggingFace | Sync to HuggingFace
# @markdown 将修复后的标签上传，删除的空标签文件也在 HF 上删除

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
import time

SYNC_TO_HF = True  # @param {type:"boolean"}

if not SYNC_TO_HF:
    print('ℹ️  SYNC_TO_HF=False，跳过上传')
elif not fixed_files:
    print('ℹ️  没有文件需要同步')
else:
    add_ops    = []
    delete_ops = []

    for (split, name), content in fixed_files.items():
        hf_path = f'labels/{split}/{name}.txt'
        if content is None:
            # 在 HF 上也删除此文件
            delete_ops.append(CommitOperationDelete(path_in_repo=hf_path))
        else:
            add_ops.append(CommitOperationAdd(
                path_in_repo=hf_path,
                path_or_fileobj=content.encode('utf-8'),
            ))

    all_ops = add_ops + delete_ops
    print(f'📤 准备提交:')
    print(f'   修复上传 : {len(add_ops)} 个')
    print(f'   删除空标签: {len(delete_ops)} 个')
    print(f'   合计      : {len(all_ops)} 个操作\n')

    CHUNK = 50
    t0    = time.time()

    for i in range(0, len(all_ops), CHUNK):
        chunk = all_ops[i:i + CHUNK]
        api.create_commit(
            repo_id=DATASET_REPO,
            repo_type='dataset',
            operations=chunk,
            commit_message=(
                f'dataset clean: fix {len(add_ops)} labels, '
                f'delete {len(delete_ops)} empty labels '
                f'({i+1}~{min(i+CHUNK, len(all_ops))}/{len(all_ops)})'
            ),
            token=HF_TOKEN,
        )
        print(f'   ✅ 已提交 {i+1}~{min(i+CHUNK, len(all_ops))} / {len(all_ops)}')

    elapsed = time.time() - t0
    SEP = '=' * 60
    print(f'\n{SEP}')
    print(f'🎉 HuggingFace 同步完成！耗时 {elapsed:.1f} 秒')
    print(f'   查看: https://huggingface.co/datasets/{DATASET_REPO}/tree/main/labels')
    print(f'{SEP}')


# %% [markdown]
# ---
# ## 📊 Step 6 — 修复后二次验证 | Post-Fix Validation


# %%
# @title 📊 Step 6 — 修复后二次验证 | Post-Fix Validation
# @markdown 重新扫描一遍，确认所有问题已解决

print('🔍 修复后二次验证...\n')

base = Path(DATASET_LOCAL)
all_clean = True

for split in ('train', 'val'):
    img_dir = base / f'images/{split}'
    lbl_dir = base / f'labels/{split}'

    if not img_dir.exists():
        continue

    img_stems = {p.stem for p in img_dir.iterdir()
                 if p.suffix.lower() in IMG_EXTS}
    lbl_files = sorted(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
    lbl_stems = {p.stem for p in lbl_files}

    n_empty        = 0
    n_wrong_class  = 0
    n_oob          = 0
    n_bad          = 0
    n_no_label     = len(img_stems - lbl_stems)
    n_no_image     = len(lbl_stems - img_stems)

    for lbl_path in lbl_files:
        r = inspect_label_file(lbl_path)
        if r['empty']:        n_empty       += 1
        if r['wrong_class']:  n_wrong_class += 1
        if r['coord_oob']:    n_oob         += 1
        if r['bad_polygon']:  n_bad         += 1

    issues_found = n_empty + n_wrong_class + n_oob + n_bad

    SEP_S = '-' * 40
    print(f'【{split}】  图像: {len(img_stems)}  标签: {len(lbl_files)}')
    print(f'   空标签      : {n_empty}')
    print(f'   类别ID非0   : {n_wrong_class}')
    print(f'   坐标越界    : {n_oob}')
    print(f'   格式错误    : {n_bad}')
    print(f'   缺标签图像  : {n_no_label}  （仅报告，不影响训练）')
    print(f'   缺图像标签  : {n_no_image}  （仅报告）')

    if issues_found == 0:
        print(f'   ✅ {split} 数据集干净！')
    else:
        print(f'   ❌ 仍有 {issues_found} 个问题，请重新运行 Step 4')
        all_clean = False
    print()

SEP = '=' * 60
print(SEP)
if all_clean:
    print('🎉 数据集已完全清洗干净！')
    print()
    print('📌 下一步：')
    print('   1. 回到训练笔记本 colab_train.ipynb')
    print('   2. 重新运行 Step 6（训练单元格）')
    print('   3. 训练配置中 copy_paste 可以保持 0.0（更稳定）')
    print('      或改为 0.1（数据集干净后可以使用）')
else:
    print('⚠️  数据集仍有问题，请检查 Step 4 的修复逻辑')
print(SEP)
