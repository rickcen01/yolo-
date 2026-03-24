"""
批量将 MinerU 导出的 DOCX 转换为 Markdown，并整理目录结构。

执行后的目标结构：
  文献markdown/
  ├── PDF/               ← 原始 PDF 按类别归档
  │   ├── B/  C/  D/  E/
  ├── B/                 ← B 类 MD + 图片
  │   ├── B1_Redmon2016_YOLOv1/
  │   │   ├── B1_Redmon2016_YOLOv1.md
  │   │   └── images/
  │   └── ...
  ├── C/  D/  E/         ← 同上
"""

import os
import re
import shutil
import mammoth
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────
SRC_DIR  = Path(r"D:\downloads\MinerU_Batch_Export_20260322204329")
BASE_DIR = Path(r"D:\biyesheji\文献markdown")
PDF_ROOT = BASE_DIR / "PDF"          # PDF 归档根目录
CATEGORIES = {"B", "C", "D", "E"}


# ── 工具函数 ──────────────────────────────────────────────────────────
def parse_docx_name(filename: str):
    """
    从文件名提取类别和论文简称。

    示例:
      MinerU_docx_B1_Redmon2016_YOLOv1_2035698857848795136.docx
      → category="B", paper_name="B1_Redmon2016_YOLOv1"
    """
    stem = Path(filename).stem                          # 去掉 .docx
    # 去掉 MinerU_docx_ 前缀
    stem = re.sub(r"^MinerU_docx_", "", stem)
    # 去掉末尾长数字时间戳  _2035698857848795136
    stem = re.sub(r"_\d{10,}$", "", stem)
    category = stem[0].upper() if stem else None
    return category, stem                               # e.g. ("B", "B1_Redmon2016_YOLOv1")


def convert_one(docx_path: Path, out_dir: Path, paper_name: str):
    """将单个 DOCX 转换为 MD，图片保存到 out_dir/images/。"""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    count = [0]

    def handle_image(image):
        count[0] += 1
        ext = image.content_type.split("/")[-1]
        ext = {"jpeg": "jpg", "x-emf": "emf", "x-wmf": "wmf"}.get(ext, ext)
        img_name = f"{count[0]}.{ext}"
        img_path = images_dir / img_name
        with image.open() as data:
            img_path.write_bytes(data.read())
        return {"src": f"images/{img_name}"}

    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_markdown(
            f,
            convert_image=mammoth.images.img_element(handle_image),
        )

    md_path = out_dir / f"{paper_name}.md"
    md_path.write_text(result.value, encoding="utf-8")

    return count[0], result.messages


# ── Step 1：把现有 PDF 从 B/C/D/E 移到 PDF/{category}/ ────────────────
def move_pdfs():
    print("=" * 62)
    print("  Step 1：归档原始 PDF → 文献markdown/PDF/{类别}/")
    print("=" * 62)
    moved = 0
    for cat in CATEGORIES:
        src_folder = BASE_DIR / cat
        dst_folder = PDF_ROOT / cat
        if not src_folder.exists():
            continue
        pdfs = sorted(src_folder.glob("*.pdf"))
        if not pdfs:
            continue
        dst_folder.mkdir(parents=True, exist_ok=True)
        for pdf in pdfs:
            dst = dst_folder / pdf.name
            if dst.exists():
                print(f"  [跳过] {cat}/{pdf.name}  已存在于 PDF/{cat}/")
                continue
            shutil.move(str(pdf), str(dst))
            print(f"  [移动] {cat}/{pdf.name}  →  PDF/{cat}/{pdf.name}")
            moved += 1
    print(f"  共移动 {moved} 个 PDF 文件\n")


# ── Step 2：清理误建的旧转换目录（含 MinerU_docx_ 前缀的文件夹）────────
def cleanup_old_dirs():
    print("=" * 62)
    print("  Step 2：清理旧的错误输出目录")
    print("=" * 62)
    removed = 0
    for item in BASE_DIR.iterdir():
        if item.is_dir() and item.name.startswith("MinerU_docx_"):
            shutil.rmtree(item)
            print(f"  [删除] {item.name}/")
            removed += 1
    if removed == 0:
        print("  无需清理")
    print()


# ── Step 3：批量转换所有 DOCX ─────────────────────────────────────────
def convert_all():
    print("=" * 62)
    print("  Step 3：批量转换 DOCX → MD + 图片")
    print("=" * 62)

    docx_files = sorted(SRC_DIR.glob("MinerU_docx_*.docx"))
    if not docx_files:
        print(f"  [错误] 在 {SRC_DIR} 中未找到 DOCX 文件")
        return

    total   = len(docx_files)
    ok      = 0
    skipped = 0
    errors  = []

    for idx, docx_path in enumerate(docx_files, 1):
        category, paper_name = parse_docx_name(docx_path.name)

        if category not in CATEGORIES:
            print(f"  [{idx:02d}/{total}] [跳过] 无法识别类别: {docx_path.name}")
            skipped += 1
            continue

        out_dir  = BASE_DIR / category / paper_name
        md_file  = out_dir / f"{paper_name}.md"

        # 已存在且 MD 大于 1 KB 则跳过
        if md_file.exists() and md_file.stat().st_size > 1024:
            size_kb = md_file.stat().st_size / 1024
            print(f"  [{idx:02d}/{total}] [跳过] {paper_name}.md 已存在 ({size_kb:.0f} KB)")
            skipped += 1
            ok += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{idx:02d}/{total}] 转换: {paper_name}", end="", flush=True)

        try:
            img_count, messages = convert_one(docx_path, out_dir, paper_name)
            md_size = md_file.stat().st_size / 1024
            print(f"  →  {category}/{paper_name}/  "
                  f"({md_size:.0f} KB MD, {img_count} 张图片)")
            if messages:
                for m in messages[:3]:
                    print(f"       [WARN] {m}")
            ok += 1
        except Exception as e:
            print(f"  [ERR] {e}")
            errors.append((paper_name, str(e)))

    print()
    print(f"  转换结果：成功 {ok} / 共 {total}（跳过 {skipped}，"
          f"失败 {len(errors)}）")
    if errors:
        print("  失败列表：")
        for name, err in errors:
            print(f"    {name}: {err}")
    print()


# ── Step 4：打印最终目录树（两级）────────────────────────────────────
def print_tree():
    print("=" * 62)
    print("  Step 4：最终目录结构预览")
    print("=" * 62)
    for cat in ["PDF"] + sorted(CATEGORIES):
        folder = BASE_DIR / cat
        if not folder.exists():
            continue
        items = sorted(folder.iterdir())
        sub_count = len(items)
        print(f"\n  文献markdown/{cat}/  ({sub_count} 个子项)")
        for item in items[:6]:          # 最多显示 6 个
            if item.is_dir():
                sub_items = list(item.iterdir())
                md_files  = [x for x in sub_items if x.suffix == ".md"]
                img_dir   = item / "images"
                img_count = len(list(img_dir.iterdir())) if img_dir.exists() else 0
                md_size   = (md_files[0].stat().st_size / 1024
                             if md_files else 0)
                print(f"    {item.name}/  "
                      f"({md_size:.0f} KB MD, {img_count} 图)")
            else:
                print(f"    {item.name}  ({item.stat().st_size/1024:.0f} KB)")
        if sub_count > 6:
            print(f"    ... 还有 {sub_count - 6} 个")
    print()


# ── 主程序 ────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  文献 DOCX → MD 批量转换 & 目录整理工具                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  来源: {SRC_DIR}")
    print(f"  输出: {BASE_DIR}")
    print()

    move_pdfs()
    cleanup_old_dirs()
    convert_all()
    print_tree()

    print("=" * 62)
    print("  全部完成！")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
