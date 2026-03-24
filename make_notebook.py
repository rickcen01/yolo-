import json
import re
import sys

# 支持命令行参数: python make_notebook.py input.py output.ipynb
# 不传参数则默认转换 colab_auto_label.py
if len(sys.argv) >= 3:
    INPUT_PY  = sys.argv[1]
    OUTPUT_NB = sys.argv[2]
elif len(sys.argv) == 2:
    INPUT_PY  = sys.argv[1]
    OUTPUT_NB = INPUT_PY.replace('.py', '.ipynb')
else:
    INPUT_PY  = 'colab_auto_label.py'
    OUTPUT_NB = 'colab_auto_label.ipynb'

with open(INPUT_PY, 'r', encoding='utf-8') as f:
    raw = f.read()

# Split on cell boundaries (# %% or # %% [markdown])
# The first chunk is the file header comment - skip it
chunks = re.split(r'\n(?=# %%)', raw)

cells = []

for chunk in chunks[1:]:  # chunks[0] is the file header comment block, skip it
    lines = chunk.split('\n')

    # Identify cell type from the marker line
    marker = lines[0] if lines else ''
    is_markdown = '[markdown]' in marker

    # Content starts after the marker line
    content_lines = lines[1:]

    # Strip leading blank lines
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)

    # Strip trailing blank lines
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()

    if not content_lines:
        continue

    if is_markdown:
        # Strip the leading '# ' or lone '#' from each line
        md_lines = []
        for line in content_lines:
            if line.startswith('# '):
                md_lines.append(line[2:])
            elif line.rstrip() == '#':
                md_lines.append('')
            else:
                # Preserve lines that don't start with # (shouldn't happen, but safety)
                md_lines.append(line)

        # Build source array: every line except the last gets a trailing \n
        source = [l + '\n' for l in md_lines[:-1]] + ([md_lines[-1]] if md_lines else [])

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        })

    else:
        # Code cell
        # Replace get_ipython().system('...') with the shell shorthand for clarity
        # (both work in Colab; keeping as-is is fine)
        source = [l + '\n' for l in content_lines[:-1]] + ([content_lines[-1]] if content_lines else [])

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "cellView": "form"
            },
            "outputs": [],
            "source": source
        })

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = OUTPUT_NB
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f'✅ 转换完成！共生成 {len(cells)} 个单元格')
print(f'📄 输入文件: {INPUT_PY}')
print(f'📄 输出文件: {out_path}')
print()
print('单元格列表:')
for i, cell in enumerate(cells):
    ctype = cell['cell_type']
    first_line = cell['source'][0].strip() if cell['source'] else '(空)'
    print(f'  [{i+1:02d}] {ctype:<10}  {first_line[:70]}')
