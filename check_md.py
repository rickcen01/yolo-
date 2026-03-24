import re

path = r"D:\biyesheji\文献markdown\_pandoc_test\C1_He2016_ResNet.md"

with open(path, encoding="utf-8") as f:
    content = f.read()

lines = content.split("\n")

# 找含数学相关符号的行
keywords = ["$", "frac", "sum", "_{", "^{", "\\mathbf", "\\times", "\\sigma", "\\alpha"]
math_lines = []
for i, line in enumerate(lines):
    for kw in keywords:
        if kw in line:
            math_lines.append((i + 1, line))
            break

print("=== 含数学符号的行（前10条）===")
for lineno, line in math_lines[:10]:
    print(f"  L{lineno}: {line[:120]}")

print()

# 找表格
table_lines = [(i + 1, l) for i, l in enumerate(lines) if l.strip().startswith("|")]
print(f"=== 表格行数: {len(table_lines)} ===")
for lineno, line in table_lines[:6]:
    print(f"  L{lineno}: {line[:100]}")

print()

# 找图片
img_refs = [(i + 1, l) for i, l in enumerate(lines) if "![" in l]
print(f"=== 图片引用: {len(img_refs)} 处 ===")
for lineno, line in img_refs[:5]:
    print(f"  L{lineno}: {line[:100]}")

print()

# 显示第60-100行（正文开始，通常有公式）
print("=== 正文片段（L60~L100）===")
for i, line in enumerate(lines[59:99], start=60):
    print(f"  L{i}: {line[:120]}")

print()
print(f"=== 文件大小: {len(content) / 1024:.1f} KB，共 {len(lines)} 行 ===")
