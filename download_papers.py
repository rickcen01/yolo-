import os
import ssl
import time
import urllib.request
import urllib.error

# ── 下载目标定义 ──────────────────────────────────────────────
PAPERS = {
    "B": [
        ("1506.02640", "B1_Redmon2016_YOLOv1"),
        ("1804.02767", "B2_Redmon2018_YOLOv3"),
        ("2207.02696", "B4_Wang2023_YOLOv7"),
        ("2402.13616", "B6_Wang2024_YOLOv9"),
        ("2405.14458", "B7_Wang2024_YOLOv10"),
        ("1506.01497", "B8_Ren2015_FasterRCNN"),
        ("1512.02325", "B9_Liu2016_SSD"),
        ("1708.02002", "B10_Lin2017_RetinaNet"),
        ("2005.12872", "B11_Carion2020_DETR"),
        ("2304.08069", "B12_Lv2024_RTDETR"),
    ],
    "C": [
        ("1512.03385", "C1_He2016_ResNet"),
        ("1905.11946", "C2_Tan2019_EfficientNet"),
        ("1801.04381", "C3_Sandler2018_MobileNetV2"),
        ("1612.03144", "C4_Lin2017_FPN"),
        ("1807.06521", "C5_Woo2018_CBAM"),
        ("1709.01507", "C6_Hu2018_SENet"),
    ],
}

BASE_DIR = r"D:\biyesheji\文献markdown"

# arXiv 两个端点依次尝试
PDF_URLS = [
    "https://arxiv.org/pdf/{arxiv_id}",
    "https://export.arxiv.org/pdf/{arxiv_id}",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 创建跳过 SSL 验证的上下文（仅用于下载学术论文，无安全风险）
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def download_pdf(arxiv_id: str, save_path: str) -> bool:
    """尝试从 arXiv 下载 PDF，成功返回 True，失败返回 False。"""
    for url_template in PDF_URLS:
        url = url_template.format(arxiv_id=arxiv_id)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as resp:
                content = resp.read()
            # 简单校验：PDF 文件以 %PDF 开头
            if not content.startswith(b"%PDF"):
                print(f"  [警告] 返回内容不是 PDF，跳过此端点: {url}")
                continue
            with open(save_path, "wb") as f:
                f.write(content)
            return True
        except urllib.error.HTTPError as e:
            print(f"  [HTTP错误] {e.code} {e.reason} — {url}")
        except urllib.error.URLError as e:
            print(f"  [连接错误] {e.reason} — {url}")
        except Exception as e:
            print(f"  [未知错误] {type(e).__name__}: {e} — {url}")
        time.sleep(2)
    return False


def main():
    total = sum(len(v) for v in PAPERS.values())
    done = 0
    failed = []

    for category, entries in PAPERS.items():
        folder = os.path.join(BASE_DIR, category)
        os.makedirs(folder, exist_ok=True)
        print(f"\n{'='*58}")
        print(f"  开始下载 {category} 类论文（共 {len(entries)} 篇）")
        print(f"{'='*58}")

        for arxiv_id, filename in entries:
            save_path = os.path.join(folder, f"{filename}.pdf")

            # 已存在且大小正常则跳过
            if os.path.exists(save_path) and os.path.getsize(save_path) > 10_000:
                print(f"  [跳过] {filename}.pdf 已存在")
                done += 1
                continue

            print(f"  [下载] {arxiv_id}  →  {filename}.pdf  ...", end="", flush=True)
            success = download_pdf(arxiv_id, save_path)

            if success:
                size_kb = os.path.getsize(save_path) / 1024
                print(f"  完成 ({size_kb:.0f} KB)")
                done += 1
            else:
                print(f"  失败")
                failed.append((category, arxiv_id, filename))
                if os.path.exists(save_path):
                    os.remove(save_path)

            # 礼貌性延迟，避免被 arXiv 限速
            time.sleep(4)

    # ── 汇总报告 ─────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  下载完成：{done} / {total} 篇成功")
    if failed:
        print(f"\n  以下 {len(failed)} 篇下载失败，请手动下载：")
        for cat, aid, name in failed:
            print(f"    [{cat}] {name}")
            print(f"          https://arxiv.org/abs/{aid}")
    else:
        print("  全部下载成功！")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
