"""
用 pandoc 批量重新转换 25 篇 DOCX → Markdown
完整保留：LaTeX 公式 ($...$ / $$...$$)、图片、Markdown 表格

目标结构（覆盖原 mammoth 转换结果）：
  文献markdown/
  ├── B/B1_Redmon2016_YOLOv1/
  │   ├── B1_Redmon2016_YOLOv1.md   ← LaTeX 公式 + 图片引用
  │   └── media/                     ← 提取的图片
  ├── C/ D/ E/ ...（同上）
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────
SRC_DIR  = Path(r"D:\downloads\MinerU_Batch_Export_20260322204329")
BASE_DIR = Path(r"D:\biyesheji\文献markdown")
CATEGORIES = {"B", "C", "D", "E"}

# ── pandoc 可执行文件路径（自动探测，找不到则报错）───────────────────
def find_pandoc() -> str:
    for candidate in ["pandoc", "pandoc.exe"]:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                ver = result.stdout.splitlines()[0]
                print(f"  使用 pandoc: {ver}")
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    sys.exit("[错误] 未找到 pandoc，请先安装：https://pandoc.org/installing.html")


# ── 文件名解析 ────────────────────────────────────────────────────────
def parse_docx_name(filename: str):
    """
    MinerU_docx_B1_Redmon2016_YOLOv1_2035698857848795136.docx
    → category="B", paper_name="B1_Redmon2016_YOLOv1"
    """
    stem = Path(filename).stem
    stem = re.sub(r"^MinerU_docx_", "", stem)
    stem = re.sub(r"_\d{10,}$", "", stem)
    category = stem[0].upper() if stem else None
    return category, stem


# ── 单文件转换 ────────────────────────────────────────────────────────
def convert_one(pandoc_bin: str, docx_path: Path, out_dir: Path, paper_name: str) -> dict:
    """
    用 pandoc 将 docx 转换为 markdown。
    图片提取到 out_dir/media/（pandoc 默认行为）。
    返回统计信息字典。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{paper_name}.md"

    # 删除旧的 mammoth 生成文件（保留 media/ 外的内容先清理）
    old_images = out_dir / "images"
    if old_images.exists():
        shutil.rmtree(old_images)

    cmd = [
        pandoc_bin,
        str(docx_path),
        "-o", str(md_path),
        # 提取图片到 out_dir/media/（pandoc 自动更新 MD 中的引用路径）
        f"--extract-media={out_dir}",
        # 输出格式：标准 markdown + LaTeX 公式 + pipe 表格 + 原始 HTML 兜底
        "-t", "markdown+tex_math_dollars+pipe_tables+raw_html",
        # 不自动折行（保持公式完整）
        "--wrap=none",

    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pandoc 非零返回码")

    # ── 统计转换结果 ──────────────────────────────────────────────────
    md_text    = md_path.read_text(encoding="utf-8", errors="replace")
    md_size_kb = md_path.stat().st_size / 1024

    # 图片：pandoc 提取到 out_dir/media/
    media_dir = out_dir / "media"
    img_count = len(list(media_dir.iterdir())) if media_dir.exists() else 0

    # 公式：行内 $...$ 和块级 $$...$$
    inline_formulas  = re.findall(r"(?<!\$)\$(?!\$)[^\$\n]{1,300}?\$(?!\$)", md_text)
    display_formulas = re.findall(r"\$\$[\s\S]{1,500}?\$\$", md_text)

    # 表格：以 | 开头的行
    table_rows = [l for l in md_text.splitlines() if l.strip().startswith("|")]

    # pandoc 警告（stderr）
    warnings = [l for l in result.stderr.splitlines() if l.strip()]

    return {
        "md_size_kb"     : md_size_kb,
        "img_count"      : img_count,
        "inline_formulas": len(inline_formulas),
        "display_formulas": len(display_formulas),
        "table_rows"     : len(table_rows),
        "warnings"       : warnings,
    }


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  pandoc 批量转换：DOCX → Markdown（公式 + 图片 + 表格）      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  来源: {SRC_DIR}")
    print(f"  输出: {BASE_DIR}")
    print()

    pandoc_bin = find_pandoc()
    print()

    # 收集所有目标 DOCX
    docx_files = sorted(SRC_DIR.glob("MinerU_docx_*.docx"))
    if not docx_files:
        sys.exit(f"[错误] 在 {SRC_DIR} 中未找到 DOCX 文件")

    total   = len(docx_files)
    ok      = 0
    skipped = 0
    errors  = []

    # 汇总统计
    total_imgs     = 0
    total_inline_f = 0
    total_display_f = 0
    total_table_rows = 0

    print(f"{'='*66}")
    print(f"  共找到 {total} 个 DOCX 文件，开始转换...")
    print(f"{'='*66}")

    for idx, docx_path in enumerate(docx_files, 1):
        category, paper_name = parse_docx_name(docx_path.name)

        if category not in CATEGORIES:
            print(f"  [{idx:02d}/{total}] [跳过] 无法识别类别: {docx_path.name}")
            skipped += 1
            continue

        out_dir = BASE_DIR / category / paper_name
        md_file = out_dir / f"{paper_name}.md"

        # 如果已存在且大于 1 KB，询问是否覆盖（批量时默认覆盖）
        exists = md_file.exists() and md_file.stat().st_size > 1024

        prefix = f"  [{idx:02d}/{total}]"
        action = "覆盖" if exists else "转换"
        print(f"{prefix} {action}: {paper_name}", end="", flush=True)

        try:
            stats = convert_one(pandoc_bin, docx_path, out_dir, paper_name)

            total_imgs      += stats["img_count"]
            total_inline_f  += stats["inline_formulas"]
            total_display_f += stats["display_formulas"]
            total_table_rows += stats["table_rows"]

            summary = (
                f"  → {stats['md_size_kb']:.0f}KB MD"
                f"  | 图片 {stats['img_count']}"
                f"  | 行内公式 {stats['inline_formulas']}"
                f"  | 块公式 {stats['display_formulas']}"
                f"  | 表格行 {stats['table_rows']}"
            )
            print(summary)

            if stats["warnings"]:
                for w in stats["warnings"][:2]:
                    print(f"         [W] {w[:100]}")

            ok += 1

        except subprocess.TimeoutExpired:
            print(f"  [超时]")
            errors.append((paper_name, "pandoc 超时（>120s）"))
        except Exception as e:
            print(f"  [ERR] {e}")
            errors.append((paper_name, str(e)))

    # ── 清理测试目录 ──────────────────────────────────────────────────
    test_dir = BASE_DIR / "_pandoc_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n  [清理] 删除测试目录 _pandoc_test/")

    # ── 汇总报告 ──────────────────────────────────────────────────────
    print()
    print(f"{'='*66}")
    print(f"  转换完成：{ok} / {total} 成功"
          f"（跳过 {skipped}，失败 {len(errors)}）")
    print(f"  {'─'*60}")
    print(f"  全部论文合计：")
    print(f"    图片提取：  {total_imgs} 张")
    print(f"    行内公式：  {total_inline_f} 处  ($...$)")
    print(f"    块级公式：  {total_display_f} 处  ($$...$$)")
    print(f"    表格行数：  {total_table_rows} 行")

    if errors:
        print(f"\n  失败列表（{len(errors)} 篇）：")
        for name, err in errors:
            print(f"    {name}")
            print(f"      原因: {err}")

    print(f"{'='*66}")
    print()

    # ── 目录结构预览 ──────────────────────────────────────────────────
    print("最终目录结构预览：")
    for cat in sorted(CATEGORIES):
        folder = BASE_DIR / cat
        if not folder.exists():
            continue
        items = [x for x in sorted(folder.iterdir()) if x.is_dir()]
        print(f"\n  文献markdown/{cat}/  （{len(items)} 篇）")
        for item in items:
            md_files = list(item.glob("*.md"))
            media    = item / "media"
            imgs     = len(list(media.iterdir())) if media.exists() else 0
            md_kb    = md_files[0].stat().st_size / 1024 if md_files else 0
            print(f"    {item.name}/  ({md_kb:.0f}KB, {imgs}张图)")
    print()


if __name__ == "__main__":
    main()
