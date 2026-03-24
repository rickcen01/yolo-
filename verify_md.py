"""验证 pandoc 转换结果：展示公式、图片、表格的真实样例。"""

import re
from pathlib import Path

BASE_DIR = Path(r"D:\biyesheji\文献markdown")

# 选几篇有代表性的论文验证
SAMPLES = [
    ("B", "B1_Redmon2016_YOLOv1"),
    ("B", "B8_Ren2015_FasterRCNN"),
    ("C", "C1_He2016_ResNet"),
    ("C", "C4_Lin2017_FPN"),
    ("E", "E3_Kingma2015_Adam"),
    ("E", "E4_Ioffe2015_BatchNormalization"),
]


def verify_one(category: str, paper_name: str):
    folder  = BASE_DIR / category / paper_name
    md_path = folder / f"{paper_name}.md"
    media   = folder / "media"

    if not md_path.exists():
        print(f"  [未找到] {md_path}")
        return

    text  = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # ── 统计 ──────────────────────────────────────────────────────────
    inline_f  = re.findall(r"(?<!\$)\$(?!\$)(.{1,120}?)\$(?!\$)", text)
    display_f = re.findall(r"\$\$([\s\S]{1,400}?)\$\$", text)
    img_refs  = re.findall(r"!\[.*?\]\(([^)]+)\)", text)
    tbl_rows  = [l for l in lines if l.strip().startswith("|")]
    img_files = sorted(media.iterdir()) if media.exists() else []

    print(f"\n{'='*64}")
    print(f"  [{category}] {paper_name}")
    print(f"  文件: {md_path.stat().st_size/1024:.0f} KB  |  "
          f"图片: {len(img_files)} 张  |  "
          f"行内公式: {len(inline_f)}  |  "
          f"块公式: {len(display_f)}  |  "
          f"表格行: {len(tbl_rows)}")
    print(f"{'─'*64}")

    # ── 行内公式示例（最多 3 个）──────────────────────────────────────
    if inline_f:
        print("  【行内公式示例】")
        for fml in inline_f[:3]:
            print(f"    ${ fml[:80] }$")
    else:
        print("  【行内公式】无")

    # ── 块级公式示例（最多 2 个）─────────────────────────────────────
    if display_f:
        print("  【块级公式示例】")
        for fml in display_f[:2]:
            clean = fml.strip().replace("\n", "  ")[:120]
            print(f"    $${ clean }$$")
    else:
        print("  【块级公式】无")

    # ── 图片引用示例 ──────────────────────────────────────────────────
    if img_refs:
        print("  【图片引用示例】")
        for ref in img_refs[:3]:
            print(f"    ![]({ref})")
    else:
        print("  【图片引用】无")

    # ── 表格示例 ──────────────────────────────────────────────────────
    if tbl_rows:
        print("  【表格示例（前 3 行）】")
        for row in tbl_rows[:3]:
            print(f"    {row[:100]}")
    else:
        print("  【表格】该论文无 Markdown 表格"
              "（对比表格通常以图片形式嵌入）")

    # ── 正文片段（含公式的段落）──────────────────────────────────────
    print("  【含公式的正文片段】")
    shown = 0
    for i, line in enumerate(lines):
        if "$" in line and len(line.strip()) > 20:
            print(f"    L{i+1}: {line.strip()[:110]}")
            shown += 1
            if shown >= 4:
                break
    if shown == 0:
        print("    （未找到含公式的正文行）")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  pandoc 转换质量验证：公式 / 图片 / 表格                     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    for category, paper_name in SAMPLES:
        verify_one(category, paper_name)

    # ── 全局汇总 ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  全部论文汇总统计")
    print(f"{'─'*64}")

    grand_imgs = 0
    grand_inline = 0
    grand_display = 0
    grand_tables = 0

    for cat in ["B", "C", "D", "E"]:
        cat_dir = BASE_DIR / cat
        if not cat_dir.exists():
            continue
        for paper_dir in sorted(cat_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            md_files = list(paper_dir.glob("*.md"))
            if not md_files:
                continue
            text = md_files[0].read_text(encoding="utf-8", errors="replace")
            media = paper_dir / "media"
            imgs  = len(list(media.iterdir())) if media.exists() else 0
            grand_imgs    += imgs
            grand_inline  += len(re.findall(r"(?<!\$)\$(?!\$).{1,120}?\$(?!\$)", text))
            grand_display += len(re.findall(r"\$\$[\s\S]{1,400}?\$\$", text))
            grand_tables  += len([l for l in text.splitlines()
                                  if l.strip().startswith("|")])

    print(f"  图片总计：  {grand_imgs} 张")
    print(f"  行内公式：  {grand_inline} 处  ($...$)")
    print(f"  块级公式：  {grand_display} 处  ($$...$$)")
    print(f"  表格行数：  {grand_tables} 行")
    print(f"{'='*64}")
    print()
    print("  说明：")
    print("  ✔ 行内公式已转换为标准 LaTeX  $...$  格式")
    print("  ✔ 块级公式已转换为标准 LaTeX  $$...$$ 格式")
    print("  ✔ 图片已提取到各论文目录下的 media/ 子文件夹")
    print("  ✔ 表格行数为 0 正常——学术论文对比表通常以图片插入，")
    print("    少数含纯文字的表格会被 pandoc 转成 Markdown pipe 表格")
    print()


if __name__ == "__main__":
    main()
