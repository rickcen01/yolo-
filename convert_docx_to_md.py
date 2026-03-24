"""批量将 Word 文档转换为 Markdown，提取图片到子文件夹。"""

import mammoth
import os
import re
from pathlib import Path

# 源文件目录
SRC_DIR = Path(r"D:\downloads\MinerU_Batch_Export_20260322204329")

# 输出根目录
OUT_DIR = Path(r"D:\biyesheji\文献markdown")

# 要转换的文件列表
DOCX_FILES = [

    "MinerU_docx_B1_Redmon2016_YOLOv1_2035698857848795136.docx",
]


def clean_folder_name(filename: str) -> str:
    """从文件名生成简洁的文件夹名，去掉时间戳后缀。"""
    name = Path(filename).stem
    # 去掉 _pdf_20260316_224207 或 _epub_20260316_224506 这类后缀
    name = re.sub(r"_(pdf|epub)_\d{8}_\d{6}$", "", name)
    return name


def convert_one(docx_path: Path, output_dir: Path, folder_name: str):
    """转换单个 docx 文件为 markdown，提取图片。"""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_count = [0]

    def handle_image(image):
        image_count[0] += 1
        ext = image.content_type.split("/")[-1]
        if ext == "jpeg":
            ext = "jpg"
        elif ext == "x-emf":
            ext = "emf"
        elif ext == "x-wmf":
            ext = "wmf"
        img_filename = f"{image_count[0]}.{ext}"
        img_path = images_dir / img_filename
        with image.open() as img_data:
            img_path.write_bytes(img_data.read())
        return {"src": f"images/{img_filename}"}

    with open(docx_path, "rb") as f:
        result = mammoth.convert_to_markdown(
            f,
            convert_image=mammoth.images.img_element(handle_image),
        )

    md_path = output_dir / f"{folder_name}.md"
    md_path.write_text(result.value, encoding="utf-8")

    return image_count[0], result.messages


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUT_DIR}")
    print()

    for filename in DOCX_FILES:
        docx_path = SRC_DIR / filename
        if not docx_path.exists():
            print(f"[跳过] 文件不存在: {docx_path}")
            continue

        folder_name = clean_folder_name(filename)
        output_dir = OUT_DIR / folder_name

        print(f"转换: {filename}")
        print(f"  → {output_dir}")

        try:
            img_count, messages = convert_one(docx_path, output_dir, folder_name)
            file_size = (output_dir / f"{folder_name}.md").stat().st_size
            print(f"  [OK] 完成 | 图片: {img_count} 张 | MD大小: {file_size / 1024:.1f} KB")
            if messages:
                for msg in messages[:5]:
                    print(f"  [WARN] {msg}")
        except Exception as e:
            print(f"  [ERR] 错误: {e}")

        print()

    print("全部完成！")


if __name__ == "__main__":
    main()
