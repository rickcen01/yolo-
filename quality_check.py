"""
严格质量检查：验证 pandoc 转换后 MD 文件的公式、图片、路径正确性
检查项目：
  1. 图片文件是否存在且未损坏（检查文件头魔术字节）
  2. 公式是否有乱码、字符拆散（如 \text{C\ l\ a\ s\ s}）
  3. 图片路径是否为绝对路径（可能导致移植问题）
  4. 公式括号是否匹配
  5. 列出真实公式样例供人工抽查
"""

import re
import os
import struct
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"D:\biyesheji\文献markdown")
CATEGORIES = ["B", "C", "D", "E"]

# 图片魔术字节签名
IMAGE_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG":      "PNG",
    b"GIF8":         "GIF",
    b"RIFF":         "WEBP/BMP",
    b"\x00\x00\x01\x00": "ICO",
    b"BM":           "BMP",
}

# 已知的公式乱码模式
BAD_FORMULA_PATTERNS = [
    # 字母被拆散：\text{C\ l\ a\ s\ s}
    (r"\\text\{([A-Za-z](?:\ [A-Za-z]){2,})\}", "字母拆散（\\text 内字符间多余空格）"),
    # 连续多个反斜杠空格
    (r"(?:\\ ){3,}", "连续多个 \\ 空格"),
    # 空公式 $$ 或 $ $
    (r"\$\s*\$", "空公式"),
    # 双重转义的美元符 \$\\
    (r"\\\$\\\\", "双重转义美元符"),
    # 未闭合的 \begin 没有对应 \end
    # （简单检测：\begin 数 != \end 数）
]


def check_image_file(img_path: Path) -> tuple[bool, str]:
    """检查图片文件是否存在且有效。"""
    if not img_path.exists():
        return False, "文件不存在"
    size = img_path.stat().st_size
    if size < 100:
        return False, f"文件过小（{size} 字节），可能已损坏"
    try:
        header = img_path.read_bytes()[:8]
    except Exception as e:
        return False, f"读取失败: {e}"
    for magic, fmt in IMAGE_MAGIC.items():
        if header[:len(magic)] == magic:
            return True, fmt
    # EMF/WMF 文件头不在列表里，但大小正常就接受
    return True, "未知格式（可能是 EMF/WMF，通常可接受）"


def check_formula_issues(formula: str) -> list[str]:
    """检查单个公式内容，返回问题列表。"""
    issues = []
    for pattern, desc in BAD_FORMULA_PATTERNS:
        if re.search(pattern, formula):
            issues.append(desc)
    # 检查括号匹配（花括号）
    depth = 0
    for ch in formula:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            issues.append("花括号不匹配（多余的 }）")
            break
    if depth > 0:
        issues.append(f"花括号不匹配（未闭合的 {{ ，差 {depth} 个 }}）")
    # 检查 \begin / \end 配对
    begins = len(re.findall(r"\\begin\{", formula))
    ends = len(re.findall(r"\\end\{", formula))
    if begins != ends:
        issues.append(f"\\begin({{)={begins} 与 \\end({{)={ends} 不匹配")
    return issues


def analyze_paper(category: str, paper_dir: Path) -> dict:
    """分析单篇论文的 MD 文件，返回检查结果。"""
    result = {
        "name":           paper_dir.name,
        "category":       category,
        "md_exists":      False,
        "md_size_kb":     0,
        "images_total":   0,
        "images_ok":      0,
        "images_bad":     [],
        "abs_path_imgs":  0,      # 图片使用绝对路径的数量
        "rel_path_imgs":  0,
        "inline_formulas": 0,
        "display_formulas": 0,
        "formula_issues": [],     # [(公式片段, [问题描述])]
        "sample_display": [],     # 块公式样例（供人工抽查）
        "sample_inline":  [],     # 行内公式样例
    }

    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        return result

    md_path = md_files[0]
    result["md_exists"] = True
    result["md_size_kb"] = md_path.stat().st_size / 1024

    text = md_path.read_text(encoding="utf-8", errors="replace")

    # ── 图片检查 ─────────────────────────────────────────────────────
    img_refs = re.findall(r"!\[.*?\]\(([^)]+)\)", text)
    for ref in img_refs:
        ref_clean = ref.split("{")[0].strip()  # 去掉 {width=...} 等属性
        # 绝对路径 vs 相对路径
        if re.match(r"^[A-Za-z]:\\", ref_clean) or ref_clean.startswith("/"):
            result["abs_path_imgs"] += 1
            img_path = Path(ref_clean)
        else:
            result["rel_path_imgs"] += 1
            img_path = paper_dir / ref_clean

        result["images_total"] += 1
        ok, fmt = check_image_file(img_path)
        if ok:
            result["images_ok"] += 1
        else:
            result["images_bad"].append((ref_clean, fmt))

    # ── 公式检查 ─────────────────────────────────────────────────────
    # 块级公式
    display_formulas = re.findall(r"\$\$([\s\S]{1,600}?)\$\$", text)
    result["display_formulas"] = len(display_formulas)
    for fml in display_formulas:
        issues = check_formula_issues(fml)
        if issues:
            result["formula_issues"].append((fml.strip()[:80], issues))
        if len(result["sample_display"]) < 3:
            result["sample_display"].append(fml.strip()[:100])

    # 行内公式
    inline_formulas = re.findall(r"(?<!\$)\$(?!\$)([\s\S]{1,200}?)\$(?!\$)", text)
    result["inline_formulas"] = len(inline_formulas)
    for fml in inline_formulas:
        issues = check_formula_issues(fml)
        if issues:
            result["formula_issues"].append((fml.strip()[:80], issues))
        if len(result["sample_inline"]) < 3:
            result["sample_inline"].append(fml.strip()[:80])

    return result


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Markdown 转换质量严格检查                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  检查目录: {BASE_DIR}")
    print()

    all_results = []
    category_stats = defaultdict(lambda: {
        "papers": 0, "imgs_ok": 0, "imgs_bad": 0,
        "formula_issues": 0, "abs_paths": 0,
        "inline": 0, "display": 0,
    })

    for cat in CATEGORIES:
        cat_dir = BASE_DIR / cat
        if not cat_dir.exists():
            continue
        for paper_dir in sorted(cat_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            r = analyze_paper(cat, paper_dir)
            if not r["md_exists"]:
                continue
            all_results.append(r)
            s = category_stats[cat]
            s["papers"]   += 1
            s["imgs_ok"]  += r["images_ok"]
            s["imgs_bad"] += len(r["images_bad"])
            s["abs_paths"] += r["abs_path_imgs"]
            s["formula_issues"] += len(r["formula_issues"])
            s["inline"]   += r["inline_formulas"]
            s["display"]  += r["display_formulas"]

    # ══════════════════════════════════════════════════════════════════
    # 报告 1：逐类别概览
    # ══════════════════════════════════════════════════════════════════
    print("━" * 66)
    print("  一、各类别概览")
    print("━" * 66)
    print(f"  {'类别':<4} {'论文数':>5} {'图片OK':>7} {'图片坏':>7} "
          f"{'公式问题':>9} {'绝对路径':>9} {'行内公式':>9} {'块公式':>7}")
    print("  " + "─" * 62)
    for cat in CATEGORIES:
        s = category_stats[cat]
        if s["papers"] == 0:
            continue
        ok_flag  = "✔" if s["imgs_bad"] == 0 else "✘"
        fml_flag = "✔" if s["formula_issues"] == 0 else "⚠"
        abs_flag = "⚠" if s["abs_paths"] > 0 else "✔"
        print(f"  {cat:<4} {s['papers']:>5} "
              f"{ok_flag}{s['imgs_ok']:>5}  "
              f"{s['imgs_bad']:>7}  "
              f"{fml_flag}{s['formula_issues']:>7}  "
              f"{abs_flag}{s['abs_paths']:>7}  "
              f"{s['inline']:>9}  "
              f"{s['display']:>7}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 报告 2：问题详情
    # ══════════════════════════════════════════════════════════════════
    bad_img_papers    = [r for r in all_results if r["images_bad"]]
    fml_issue_papers  = [r for r in all_results if r["formula_issues"]]
    abs_path_papers   = [r for r in all_results if r["abs_path_imgs"] > 0]

    print("━" * 66)
    print("  二、图片问题详情")
    print("━" * 66)
    if not bad_img_papers:
        print("  ✔ 全部图片文件完好，无损坏或缺失")
    else:
        for r in bad_img_papers:
            print(f"  [{r['category']}] {r['name']}")
            for ref, reason in r["images_bad"]:
                print(f"      ✘ {Path(ref).name:<30}  原因: {reason}")
    print()

    print("━" * 66)
    print("  三、公式问题详情（⚠ 需关注）")
    print("━" * 66)
    if not fml_issue_papers:
        print("  ✔ 所有公式检查通过，无明显乱码或括号不匹配")
    else:
        for r in fml_issue_papers:
            print(f"\n  [{r['category']}] {r['name']}  "
                  f"（{len(r['formula_issues'])} 处问题）")
            shown = 0
            for fml_snippet, issues in r["formula_issues"]:
                if shown >= 5:
                    remaining = len(r["formula_issues"]) - shown
                    print(f"      ... 还有 {remaining} 处（省略）")
                    break
                print(f"    公式片段: {fml_snippet}")
                for iss in issues:
                    print(f"      ⚠ {iss}")
                shown += 1
    print()

    print("━" * 66)
    print("  四、图片路径类型（绝对路径在跨机器时会失效）")
    print("━" * 66)
    if not abs_path_papers:
        print("  ✔ 所有图片使用相对路径")
    else:
        abs_total = sum(r["abs_path_imgs"] for r in abs_path_papers)
        print(f"  ⚠ 发现 {abs_total} 处绝对路径图片引用（共 {len(abs_path_papers)} 篇论文）")
        print("  说明：绝对路径在本机正常显示，换电脑后图片无法显示")
        print("  影响：仅影响 Obsidian/Typora 预览，不影响内容阅读")
        print()
        print("  受影响的论文：")
        for r in abs_path_papers[:5]:
            print(f"    [{r['category']}] {r['name']}  "
                  f"({r['abs_path_imgs']} 处绝对路径)")
        if len(abs_path_papers) > 5:
            print(f"    ... 还有 {len(abs_path_papers)-5} 篇（共 {len(abs_path_papers)} 篇全部如此）")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 报告 3：公式样例人工抽查
    # ══════════════════════════════════════════════════════════════════
    print("━" * 66)
    print("  五、公式样例（人工抽查，每篇取前 2 条块公式）")
    print("━" * 66)
    spot_papers = [
        ("B", "B1_Redmon2016_YOLOv1"),
        ("B", "B8_Ren2015_FasterRCNN"),
        ("B", "B11_Carion2020_DETR"),
        ("C", "C1_He2016_ResNet"),
        ("C", "C5_Woo2018_CBAM"),
        ("D", "D2_Hussain2023_YOLOv1toV8Industrial"),
        ("E", "E3_Kingma2015_Adam"),
        ("E", "E4_Ioffe2015_BatchNormalization"),
    ]
    for cat, name in spot_papers:
        r = next((x for x in all_results if x["name"] == name), None)
        if not r:
            continue
        print(f"\n  [{cat}] {name}")
        if r["sample_display"]:
            for i, fml in enumerate(r["sample_display"][:2], 1):
                print(f"    块公式{i}: $${fml}$$")
        else:
            print("    （无块公式）")
        if r["sample_inline"]:
            print(f"    行内示例: ${r['sample_inline'][0]}$")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 报告 4：总结与建议
    # ══════════════════════════════════════════════════════════════════
    total_papers = len(all_results)
    total_imgs   = sum(r["images_total"] for r in all_results)
    total_ok     = sum(r["images_ok"] for r in all_results)
    total_bad    = sum(len(r["images_bad"]) for r in all_results)
    total_fml_issues = sum(len(r["formula_issues"]) for r in all_results)
    total_inline = sum(r["inline_formulas"] for r in all_results)
    total_display = sum(r["display_formulas"] for r in all_results)

    print("━" * 66)
    print("  六、总结")
    print("━" * 66)
    print(f"  论文总数:      {total_papers} 篇")
    print(f"  图片总计:      {total_imgs} 张  "
          f"（完好: {total_ok}  损坏/缺失: {total_bad}）")
    print(f"  行内公式:      {total_inline} 处")
    print(f"  块级公式:      {total_display} 处")
    print(f"  公式检查问题:  {total_fml_issues} 处")
    print()

    if total_bad == 0 and total_fml_issues == 0:
        print("  ✅ 图片和公式均无明显错误，转换质量良好")
    else:
        if total_bad > 0:
            print(f"  ❌ 有 {total_bad} 张图片损坏或缺失，需手动排查")
        if total_fml_issues > 0:
            print(f"  ⚠  有 {total_fml_issues} 处公式存在格式问题")
            print("     常见原因：Word 原文中文字被单独输入（非方程编辑器），")
            print("     此类公式在 PDF 中显示正常，但 DOCX 转 MD 时会有空格。")
            print("     建议：对比 PDF 原文核实这些公式，必要时手动修正。")

    if abs_path_papers:
        print()
        print(f"  ℹ  图片路径均为绝对路径（共 {len(abs_path_papers)} 篇）")
        print("     本机使用完全正常，若需分享给他人，运行以下命令修复：")
        print("       python fix_abs_paths.py")
    print()
    print("━" * 66)
    print()


if __name__ == "__main__":
    main()
