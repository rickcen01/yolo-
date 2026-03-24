"""
修复 pandoc 转换后 MD 文件的两类问题：
  1. 绝对路径图片引用 → 相对路径  (media/rIdXX.jpg)
  2. 字母拆散公式  \text{C\ l\ a\ s\ s} → \text{Class}
     以及 \text{i\ f} → \text{if}  等常见缩写

运行后直接覆盖原 MD 文件，原始内容自动备份到同目录 .md.bak
"""

import re
import shutil
from pathlib import Path

BASE_DIR = Path(r"D:\biyesheji\文献markdown")
CATEGORIES = ["B", "C", "D", "E"]

# ── 需要修复的 \text{...} 常见拼写（优先精确匹配）────────────────────
# 格式：(错误的 LaTeX 片段, 正确替换)
# 这些都是论文里出现的固定文字标签，拆散后逐一还原
TEXT_FIXES_EXACT = [
    # 分支条件
    (r"\\text\{i\\?\ f\}",                    r"\\text{if}"),
    (r"\\text\{o\\?\ t\\?\ h\\?\ e\\?\ r\\?\ w\\?\ i\\?\ s\\?\ e\}",
                                               r"\\text{otherwise}"),
    # 常见单词
    (r"\\text\{C\\?\ l\\?\ a\\?\ s\\?\ s\}",  r"\\text{Class}"),
    (r"\\text\{O\\?\ b\\?\ j\\?\ e\\?\ c\\?\ t\}",
                                               r"\\text{Object}"),
    (r"\\text\{t\\?\ r\\?\ u\\?\ t\\?\ h\}",  r"\\text{truth}"),
    (r"\\text\{p\\?\ r\\?\ e\\?\ d\}",        r"\\text{pred}"),
    (r"\\text\{m\\?\ a\\?\ t\\?\ c\\?\ h\}",  r"\\text{match}"),
    (r"\\text\{n\\?\ o\\?\ o\\?\ b\\?\ j\}",  r"\\text{noobj}"),
    (r"\\text\{o\\?\ b\\?\ j\}",              r"\\text{obj}"),
    (r"\\text\{r\\?\ e\\?\ s\\?\ p\}",        r"\\text{resp}"),
    (r"\\text\{l\\?\ o\\?\ c\}",              r"\\text{loc}"),
    (r"\\text\{c\\?\ l\\?\ s\}",              r"\\text{cls}"),
    (r"\\text\{c\\?\ o\\?\ n\\?\ f\}",        r"\\text{conf}"),
    (r"\\text\{b\\?\ o\\?\ x\}",              r"\\text{box}"),
    (r"\\text\{b\\?\ a\\?\ c\\?\ k\\?\ g\\?\ r\\?\ o\\?\ u\\?\ n\\?\ d\}",
                                               r"\\text{background}"),
    (r"\\text\{f\\?\ o\\?\ r\\?\ e\\?\ g\\?\ r\\?\ o\\?\ u\\?\ n\\?\ d\}",
                                               r"\\text{foreground}"),
    (r"\\text\{a\\?\ n\\?\ c\\?\ h\\?\ o\\?\ r\}",
                                               r"\\text{anchor}"),
    (r"\\text\{p\\?\ o\\?\ s\\?\ i\\?\ t\\?\ i\\?\ v\\?\ e\}",
                                               r"\\text{positive}"),
    (r"\\text\{n\\?\ e\\?\ g\\?\ a\\?\ t\\?\ i\\?\ v\\?\ e\}",
                                               r"\\text{negative}"),
    # DETR 专用
    (r"\\text\{m\\?\ a\\?\ t\\?\ c\\?\ h\\?\ e\\?\ d\}",
                                               r"\\text{matched}"),
    (r"\\text\{H\\?\ u\\?\ n\\?\ g\\?\ a\\?\ r\\?\ i\\?\ a\\?\ n\}",
                                               r"\\text{Hungarian}"),
    # ImageNet / 评测
    (r"\\text\{t\\?\ o\\?\ p\\?\ -\\?\ 5\}",  r"\\text{top-5}"),
    (r"\\text\{t\\?\ o\\?\ p\\?\ -\\?\ 1\}",  r"\\text{top-1}"),
    # BatchNorm / Adam
    (r"\\text\{B\\?\ N\}",                    r"\\text{BN}"),
    (r"\\text\{S\\?\ G\\?\ D\}",              r"\\text{SGD}"),
    (r"\\text\{R\\?\ e\\?\ L\\?\ U\}",        r"\\text{ReLU}"),
    (r"\\text\{t\\?\ r\\?\ a\\?\ i\\?\ n\}",  r"\\text{train}"),
    (r"\\text\{t\\?\ e\\?\ s\\?\ t\}",        r"\\text{test}"),
    (r"\\text\{v\\?\ a\\?\ l\}",              r"\\text{val}"),
]

# ── 通用拆散模式（兜底）：\text{X\ Y\ Z} → \text{XYZ} ────────────────
# 匹配 \text{ 内部由单字母 + `\ ` 间隔组成的内容，合并为连续字符串
SCATTERED_TEXT_RE = re.compile(
    r"\\text\{([A-Za-z0-9](?:(?:\\\ |[\ \-])[A-Za-z0-9])+)\}"
)


def fix_scattered_text(match: re.Match) -> str:
    """将 \text{C\ l\ a\ s\ s} 变成 \text{Class}（去除内部空格/反斜杠）。"""
    inner = match.group(1)
    # 去除 `\ ` 和普通空格，把字符连接起来
    fixed = re.sub(r"\\\ |\ ", "", inner)
    return rf"\text{{{fixed}}}"


def fix_abs_image_paths(text: str, md_path: Path) -> tuple[str, int]:
    """
    把绝对路径图片引用改为相对路径。

    pandoc 生成的绝对路径格式：
      ![...](D:\biyesheji\文献markdown\B\B1_xxx\media\rId11.jpg){width=...}
    目标相对路径：
      ![...](media/rId11.jpg){width=...}
    """
    count = 0
    paper_dir = md_path.parent  # MD 文件所在目录

    def replacer(m: re.Match) -> str:
        nonlocal count
        alt  = m.group(1)
        path = m.group(2)
        rest = m.group(3)  # 可能有 {width=...} 等属性

        # 只处理绝对路径（Windows 盘符 或 /开头）
        if not (re.match(r"^[A-Za-z]:[/\\]", path) or path.startswith("/")):
            return m.group(0)

        img_path = Path(path)
        try:
            # 计算相对路径
            rel = img_path.relative_to(paper_dir)
            # Windows 反斜杠统一改为正斜杠
            rel_str = rel.as_posix()
        except ValueError:
            # relative_to 失败：只保留文件名作为 media/{name}
            rel_str = f"media/{img_path.name}"

        count += 1
        return f"![{alt}]({rel_str}){rest}"

    # 匹配 ![alt](path){optional_attrs} 或 ![alt](path)
    pattern = re.compile(
        r"!\[([^\]]*)\]"          # ![alt]
        r"\(([^)]+)\)"            # (path)
        r"(\{[^}]*\})?"           # 可选的 {width=...}
    )
    new_text = pattern.sub(replacer, text)
    return new_text, count


def fix_md_file(md_path: Path) -> dict:
    """处理单个 MD 文件，返回修改统计。"""
    original = md_path.read_text(encoding="utf-8", errors="replace")
    text = original

    # ── Step 1：精确文本标签修复 ─────────────────────────────────────
    exact_fixes = 0
    for pattern, replacement in TEXT_FIXES_EXACT:
        new_text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        if n:
            text = new_text
            exact_fixes += n

    # ── Step 2：通用拆散字母修复（兜底）─────────────────────────────
    new_text, general_fixes = re.subn(SCATTERED_TEXT_RE, fix_scattered_text, text)
    if general_fixes:
        text = new_text

    # ── Step 3：绝对路径 → 相对路径 ──────────────────────────────────
    text, path_fixes = fix_abs_image_paths(text, md_path)

    total_changes = exact_fixes + general_fixes + path_fixes
    if total_changes == 0:
        return {"changed": False, "exact": 0, "general": 0, "paths": 0}

    # ── 备份原文件 ────────────────────────────────────────────────────
    bak_path = md_path.with_suffix(".md.bak")
    if not bak_path.exists():            # 已有备份就不覆盖
        shutil.copy2(md_path, bak_path)

    # ── 写入修复后的内容 ──────────────────────────────────────────────
    md_path.write_text(text, encoding="utf-8")

    return {
        "changed": True,
        "exact":   exact_fixes,
        "general": general_fixes,
        "paths":   path_fixes,
    }


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  MD 文件修复：公式字母拆散 + 图片绝对路径 → 相对路径         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  目录: {BASE_DIR}")
    print()

    total_files   = 0
    changed_files = 0
    total_exact   = 0
    total_general = 0
    total_paths   = 0

    for cat in CATEGORIES:
        cat_dir = BASE_DIR / cat
        if not cat_dir.exists():
            continue

        cat_changed = 0
        for paper_dir in sorted(cat_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            for md_path in paper_dir.glob("*.md"):
                total_files += 1
                stats = fix_md_file(md_path)

                if stats["changed"]:
                    changed_files += 1
                    cat_changed   += 1
                    total_exact   += stats["exact"]
                    total_general += stats["general"]
                    total_paths   += stats["paths"]

                    parts = []
                    if stats["exact"]:
                        parts.append(f"精确修复 {stats['exact']} 处")
                    if stats["general"]:
                        parts.append(f"通用拆散修复 {stats['general']} 处")
                    if stats["paths"]:
                        parts.append(f"路径转换 {stats['paths']} 处")
                    print(f"  [{cat}] {paper_dir.name}")
                    print(f"       {', '.join(parts)}")
                else:
                    print(f"  [{cat}] {paper_dir.name}  ← 无需修改")

        if cat_changed > 0:
            print()

    # ── 汇总报告 ──────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  处理文件:      {total_files} 个 MD 文件")
    print(f"  实际修改:      {changed_files} 个")
    print(f"  精确文本修复:  {total_exact} 处  "
          r"(\text{i\ f} → \text{if} 等)")
    print(f"  通用拆散修复:  {total_general} 处  "
          r"(自动合并 \text 内字母)")
    print(f"  路径转换:      {total_paths} 处  (绝对路径 → 相对路径)")
    print()
    print("  ✔ 原始文件已备份为 .md.bak（同目录）")
    print("  ✔ 图片路径已改为相对路径，可跨机器/跨平台使用")
    print()

    # ── 修复后二次抽查 ────────────────────────────────────────────────
    print("  二次抽查（修复后效果）：")
    spot = [
        ("B", "B1_Redmon2016_YOLOv1"),
        ("B", "B11_Carion2020_DETR"),
        ("C", "C1_He2016_ResNet"),
        ("E", "E3_Kingma2015_Adam"),
    ]
    for cat, name in spot:
        md_path = BASE_DIR / cat / name / f"{name}.md"
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        # 检查是否还有残余拆散
        remaining = re.findall(
            r"\\text\{[A-Za-z](?:(?:\\\ |\ )[A-Za-z]){2,}\}",
            text
        )
        # 检查图片是否已改为相对路径
        abs_imgs = re.findall(r"!\[.*?\]\([A-Za-z]:[/\\]", text)
        # 取一个块公式样例
        display = re.findall(r"\$\$([\s\S]{1,80}?)\$\$", text)
        sample = display[0].strip().replace("\n", " ")[:80] if display else "（无块公式）"

        print(f"\n    [{cat}] {name}")
        print(f"      残余字母拆散: {len(remaining)} 处"
              + ("  ✔" if not remaining else "  ← 仍需手动核查"))
        print(f"      绝对路径图片: {len(abs_imgs)} 处"
              + ("  ✔" if not abs_imgs else "  ← 仍有绝对路径"))
        print(f"      块公式示例: $${sample}$$")

    print()
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
