# @title 🧹 Step 8 — 数据清洗（可视化质检 + 交互删除）
# @markdown 扫描全部标注 → 展示可疑图像 + 多边形 → 勾选错误的 → 点击删除
# @markdown
# @markdown | 颜色 | 含义 |
# @markdown |------|------|
# @markdown | 🔴 红色标题 | 空标注，显示原图，判断是否真的没有气瓶 |
# @markdown | 🟠 橙色轮廓 | 点数过少，轮廓粗糙 |
# @markdown | 🔵 蓝色轮廓 | 点数过多，可能含噪声 |

import requests, os, cv2, numpy as np, base64
import ipywidgets as widgets
from IPython.display import display as ipy_display, HTML
from pathlib import Path

CHECK_EMPTY_LABELS = True   # @param {type:"boolean"}
CHECK_POINT_COUNT  = True   # @param {type:"boolean"}
MIN_POINTS         = 8      # @param {type:"integer"}
MAX_POINTS         = 500    # @param {type:"integer"}
MAX_SHOW           = 20     # @param {type:"integer"}
COLS               = 2      # @param {type:"integer"}

# ══════════════════════════════════════════════════════════════
# 阶段 1 — 扫描所有标签文件
# ══════════════════════════════════════════════════════════════
print('🔍 正在扫描标注文件...')
all_files_clean = list(api.list_repo_files(DATASET_REPO, repo_type='dataset'))

label_files_clean = sorted([
    f for f in all_files_clean
    if f.startswith('labels/train/') and f.endswith('.txt')
])
image_map = {
    os.path.splitext(os.path.basename(f))[0]: f
    for f in all_files_clean
    if f.startswith('images/train/')
    and f.lower().endswith(('.png', '.jpg', '.jpeg'))
}

label_contents = {}
empty_labels   = []
few_pts        = []
many_pts       = []
checked = errors = 0

print(f'📋 共 {len(label_files_clean)} 个标签，扫描中...\n')

for i, lbl_path in enumerate(label_files_clean):
    name = os.path.splitext(os.path.basename(lbl_path))[0]
    try:
        url = (f'https://huggingface.co/datasets/{DATASET_REPO}'
               f'/resolve/main/{lbl_path}')
        r = requests.get(url, headers={'Authorization': f'Bearer {HF_TOKEN}'},
                         timeout=30)
        r.raise_for_status()
        content = r.text.strip()
        label_contents[name] = content
        checked += 1
        if not content:
            empty_labels.append(name)
        elif CHECK_POINT_COUNT:
            n_pts = (len(content.split()) - 1) // 2
            if n_pts < MIN_POINTS:
                few_pts.append((name, n_pts))
            elif n_pts > MAX_POINTS:
                many_pts.append((name, n_pts))
    except Exception as e:
        errors += 1
    if (i + 1) % 50 == 0:
        print(f'   已扫描 {i+1}/{len(label_files_clean)}...')

SEP = '=' * 52
print(f'\n{SEP}')
print(f'  ✅ 已扫描    : {checked}  |  ❌ 读取失败: {errors}')
print(f'  🔴 空标注    : {len(empty_labels)} 个')
print(f'  🟠 点数<{MIN_POINTS}   : {len(few_pts)} 个')
print(f'  🔵 点数>{MAX_POINTS} : {len(many_pts)} 个')
print(f'{SEP}\n')

# ══════════════════════════════════════════════════════════════
# 阶段 2 — 工具函数
# ══════════════════════════════════════════════════════════════
CLEAN_TMP = Path('/tmp/clean_vis')
CLEAN_TMP.mkdir(exist_ok=True)

def fetch_img(name):
    if name not in image_map:
        return None
    dest = CLEAN_TMP / Path(image_map[name]).name
    if not dest.exists():
        url = (f'https://huggingface.co/datasets/{DATASET_REPO}'
               f'/resolve/main/{image_map[name]}')
        try:
            r = requests.get(url, headers={'Authorization': f'Bearer {HF_TOKEN}'},
                             stream=True, timeout=180)
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(1 << 17):
                    f.write(chunk)
        except Exception as e:
            print(f'❌ 下载失败 {name}: {e}')
            return None
    raw = np.fromfile(str(dest), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

def draw_poly(img, label_str, color_bgr):
    vis = img.copy()
    h, w = vis.shape[:2]
    parts = label_str.strip().split()
    if len(parts) < 7:
        return vis
    vals = [float(v) for v in parts[1:]]
    pts  = (np.array(vals).reshape(-1, 2) * [w, h]).astype(np.int32).reshape(-1, 1, 2)
    ov = vis.copy()
    cv2.fillPoly(ov, [pts], color_bgr)
    cv2.addWeighted(ov, 0.35, vis, 0.65, 0, vis)
    cv2.polylines(vis, [pts], True, color_bgr, 3)
    return vis

def to_b64(img, max_w=440):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode() if ok else ''

# ══════════════════════════════════════════════════════════════
# 阶段 3 — 逐类展示 + 生成勾选框
# ══════════════════════════════════════════════════════════════
all_checkboxes = {}   # name -> widgets.Checkbox，汇总所有类别

def show_section(entries, section_title, color_bgr):
    """下载并显示一批图像，每张图片下方有一个勾选框"""
    if not entries:
        return
    show = entries[:MAX_SHOW]

    ipy_display(HTML(
        f'<h3 style="margin:18px 0 8px 0; color:#bf360c; font-family:sans-serif">'
        f'{section_title}'
        f' &nbsp;<span style="font-weight:normal;font-size:14px;color:#777">'
        f'共 {len(entries)} 张'
        f'{"，仅展示前 " + str(MAX_SHOW) + " 张" if len(entries) > MAX_SHOW else ""}'
        f'</span></h3>'
    ))

    item_widgets = []
    for idx, (name, lbl) in enumerate(show):
        print(f'  ⬇️  下载图像 {idx+1}/{len(show)}: {name}    ', end='\r')
        img = fetch_img(name)
        if img is None:
            continue

        vis   = draw_poly(img, lbl, color_bgr) if lbl else img.copy()
        n_pts = (len(lbl.split()) - 1) // 2 if lbl else 0
        info  = f'{n_pts} 个多边形顶点' if lbl else '空标注（无多边形）'
        b64   = to_b64(vis)

        img_html = widgets.HTML(value=
            f'<div style="border:1px solid #ddd;border-radius:6px;'
            f'overflow:hidden;background:#fff;box-shadow:0 1px 3px #0001">'
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="width:100%;display:block;"/>'
            f'<div style="padding:5px 8px;font-size:11px;color:#555;'
            f'border-top:1px solid #eee;text-align:center">'
            f'<b style="color:#333">{name}</b><br>'
            f'<span style="color:#888">{info}</span>'
            f'</div></div>'
        )

        cb = widgets.Checkbox(
            value=False,
            description='标记为删除',
            indent=False,
            layout=widgets.Layout(width='auto', margin='4px 0 8px 6px')
        )
        cb.style.description_width = '0px'
        all_checkboxes[name] = cb

        item_widgets.append(widgets.VBox(
            [img_html, cb],
            layout=widgets.Layout(width='48%', margin='4px', padding='2px')
        ))

    print(' ' * 70, end='\r')

    # 排成 COLS 列
    rows_w = []
    for i in range(0, len(item_widgets), COLS):
        rows_w.append(widgets.HBox(
            item_widgets[i:i + COLS],
            layout=widgets.Layout(width='100%', flex_flow='row wrap')
        ))
    ipy_display(widgets.VBox(rows_w, layout=widgets.Layout(width='100%')))


print('🖼️  开始展示可疑图像（每张约 20~30 MB，请耐心等待）\n')

if CHECK_EMPTY_LABELS:
    if empty_labels:
        show_section([(n, '') for n in empty_labels],
                     '🔴 空标注 — 未检测到气瓶，判断是否真的没有气瓶',
                     (0, 0, 200))
    else:
        print('✅ 无空标注\n')

if CHECK_POINT_COUNT:
    if few_pts:
        show_section([(n, label_contents[n]) for n, _ in few_pts],
                     f'🟠 点数过少（< {MIN_POINTS} 点）— 判断轮廓是否贴合气瓶',
                     (0, 140, 255))
    else:
        print(f'✅ 无点数 < {MIN_POINTS} 的标注\n')

    if many_pts:
        show_section([(n, label_contents[n]) for n, _ in many_pts],
                     f'🔵 点数过多（> {MAX_POINTS} 点）— 判断是否含噪声杂点',
                     (200, 80, 0))
    else:
        print(f'✅ 无点数 > {MAX_POINTS} 的标注\n')

# ══════════════════════════════════════════════════════════════
# 阶段 4 — 底部删除面板
# ══════════════════════════════════════════════════════════════
ipy_display(HTML(
    '<hr style="margin:24px 0 12px 0; border-color:#ddd"/>'
    '<h3 style="font-family:sans-serif;color:#333">🗑️ 删除选中的标签</h3>'
    '<p style="color:#888;font-size:13px;margin:0 0 10px 0">'
    '勾选上方图像后点击下方按钮，将对应的 labels/train/xxx.txt 从 HuggingFace 删除</p>'
))

if not all_checkboxes:
    ipy_display(HTML('<p style="color:green">✅ 没有可疑的标注，无需删除。</p>'))
else:
    # 实时显示当前选中数量
    selected_label = widgets.HTML(
        value='<span style="color:#999;font-size:13px">尚未勾选任何图像</span>'
    )

    delete_btn = widgets.Button(
        description='🗑️  确认删除选中标签',
        button_style='danger',
        layout=widgets.Layout(width='220px', height='40px')
    )

    output_area = widgets.Output()

    def _refresh_count(change=None):
        selected = [n for n, cb in all_checkboxes.items() if cb.value]
        if selected:
            names_str = '、'.join(selected[:4])
            if len(selected) > 4:
                names_str += f' 等 {len(selected)} 张'
            selected_label.value = (
                f'<span style="color:#c62828;font-weight:bold">'
                f'已勾选 {len(selected)} 张：{names_str}</span>'
            )
        else:
            selected_label.value = (
                '<span style="color:#999;font-size:13px">尚未勾选任何图像</span>'
            )

    # 每个勾选框变化时都刷新计数
    for _cb in all_checkboxes.values():
        _cb.observe(_refresh_count, names='value')

    def on_delete_clicked(btn):
        with output_area:
            output_area.clear_output(wait=True)
            to_delete = [n for n, cb in all_checkboxes.items() if cb.value]

            if not to_delete:
                print('⚠️  请先勾选需要删除的图像')
                return

            delete_btn.disabled    = True
            delete_btn.description = '⏳ 删除中...'
            print(f'🗑️  正在从 HuggingFace 删除 {len(to_delete)} 个标签...\n')

            ok_list, fail_list = [], []
            for name in to_delete:
                try:
                    api.delete_file(
                        path_in_repo=f'labels/train/{name}.txt',
                        repo_id=DATASET_REPO,
                        repo_type='dataset',
                        commit_message=f'manual clean: remove {name}.txt',
                    )
                    ok_list.append(name)
                    print(f'   ✅ 已删除  labels/train/{name}.txt')
                except Exception as e:
                    fail_list.append(name)
                    print(f'   ❌ 失败    {name}: {e}')

            print(f'\n{SEP}')
            print(f'✅ 成功删除 {len(ok_list)} 个')
            if fail_list:
                print(f'❌ 失败 {len(fail_list)} 个: {fail_list}')
            print(f'{SEP}')
            print('💡 提示：被删除的图像下次运行 Step 4~6 时会重新自动标注')

            delete_btn.disabled    = False
            delete_btn.description = '🗑️  确认删除选中标签'

    delete_btn.on_click(on_delete_clicked)

    ipy_display(
        selected_label,
        widgets.HBox([delete_btn],
                     layout=widgets.Layout(margin='10px 0')),
        output_area
    )
