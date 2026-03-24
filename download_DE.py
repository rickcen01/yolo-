import os
import ssl
import time
import urllib.request
import urllib.error

# ── SSL 上下文（绕过证书验证，仅用于学术论文下载）──────────────────
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Referer": "https://www.mdpi.com/",
}

BASE_DIR = r"D:\biyesheji\文献markdown"

# ── D 类：工业视觉检测英文论文（直链下载）────────────────────────────
# 格式：(文件名, 直接PDF链接, 说明)
D_PAPERS = [
    (
        "D1_Saberironaghi2023_DefectDetectionReview.pdf",
        "https://mdpi-res.com/d_attachment/algorithms/algorithms-16-00095/"
        "article_deploy/algorithms-16-00095.pdf",
        "MDPI Algorithms 2023 | 工业缺陷检测深度学习综述",
    ),
    (
        "D2_Hussain2023_YOLOv1toV8Industrial.pdf",
        "https://mdpi-res.com/d_attachment/machines/machines-11-00677/"
        "article_deploy/machines-11-00677.pdf",
        "MDPI Machines 2023 | YOLO演进与工业缺陷检测综述",
    ),
    (
        "D3_Kim2021_ProductInspectionDeepLearning.pdf",
        "https://mdpi-res.com/d_attachment/sensors/sensors-21-05039/"
        "article_deploy/sensors-21-05039-v2.pdf",
        "MDPI Sensors 2021 | 基于深度学习的产品检测方法综述",
    ),
    (
        "D4_Prunella2023_SurfaceDefectsSurveyIEEE.pdf",
        "https://ieeexplore.ieee.org/ielx7/6287639/10005208/10113226.pdf",
        "IEEE Access 2023 | 工业表面缺陷深度学习视觉识别综述",
    ),
]

# ── E 类：开题报告补充论文（arXiv 下载）──────────────────────────────
# 格式：(文件名, arXiv ID, 说明)
E_PAPERS_ARXIV = [
    (
        "E1_Lin2014_MicrosoftCOCO.pdf",
        "1405.0312",
        "arXiv 2014 | Microsoft COCO数据集 - 通用标注格式基础",
    ),
    (
        "E2_Zoph2020_DataAugmentationObjectDetection.pdf",
        "1906.11172",
        "arXiv 2019/ECCV 2020 | 目标检测数据增强策略",
    ),
    (
        "E3_Kingma2015_Adam.pdf",
        "1412.6980",
        "arXiv 2015/ICLR 2015 | Adam优化器 - 深度学习训练优化",
    ),
    (
        "E4_Ioffe2015_BatchNormalization.pdf",
        "1502.03167",
        "arXiv 2015/ICML 2015 | 批归一化 - 模型训练稳定性",
    ),
    (
        "E5_Russakovsky2015_ImageNet.pdf",
        "1409.0575",
        "arXiv 2015 | ImageNet大规模视觉识别挑战 - 迁移学习基础",
    ),
]

ARXIV_URLS = [
    "https://arxiv.org/pdf/{arxiv_id}",
    "https://export.arxiv.org/pdf/{arxiv_id}",
]


# ── 通用下载函数 ──────────────────────────────────────────────────────
def download_url(url: str, save_path: str, retry: int = 2) -> bool:
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as resp:
                content = resp.read()
            if not content.startswith(b"%PDF"):
                print(f"    └─ [警告] 非PDF内容 (尝试 {attempt+1}/{retry})")
                time.sleep(3)
                continue
            with open(save_path, "wb") as f:
                f.write(content)
            return True
        except urllib.error.HTTPError as e:
            print(f"    └─ [HTTP {e.code}] {url}")
        except urllib.error.URLError as e:
            print(f"    └─ [连接错误] {e.reason}")
        except Exception as e:
            print(f"    └─ [错误] {type(e).__name__}: {e}")
        if attempt < retry - 1:
            time.sleep(4)
    return False


def download_arxiv(arxiv_id: str, save_path: str) -> bool:
    for url_tpl in ARXIV_URLS:
        url = url_tpl.format(arxiv_id=arxiv_id)
        if download_url(url, save_path, retry=1):
            return True
        time.sleep(3)
    return False


def skip_if_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 10_000


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    done, failed = 0, []
    total = len(D_PAPERS) + len(E_PAPERS_ARXIV)

    # ── 下载 D 类 ─────────────────────────────────────────────────────
    d_folder = os.path.join(BASE_DIR, "D")
    os.makedirs(d_folder, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  ▶ D类：工业视觉检测英文论文（共 {len(D_PAPERS)} 篇）")
    print("=" * 60)

    for filename, url, desc in D_PAPERS:
        save_path = os.path.join(d_folder, filename)
        print(f"\n  [{filename}]")
        print(f"  {desc}")

        if skip_if_exists(save_path):
            print(f"  ✔ 已存在，跳过")
            done += 1
            continue

        print(f"  → 下载中...", end="", flush=True)
        if download_url(url, save_path):
            kb = os.path.getsize(save_path) / 1024
            print(f" 完成 ({kb:.0f} KB)")
            done += 1
        else:
            print(f" 失败")
            failed.append(("D", filename, url))
            if os.path.exists(save_path):
                os.remove(save_path)

        time.sleep(4)

    # ── 下载 E 类 ─────────────────────────────────────────────────────
    e_folder = os.path.join(BASE_DIR, "E")
    os.makedirs(e_folder, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  ▶ E类：开题报告补充论文（共 {len(E_PAPERS_ARXIV)} 篇）")
    print("=" * 60)

    for filename, arxiv_id, desc in E_PAPERS_ARXIV:
        save_path = os.path.join(e_folder, filename)
        print(f"\n  [{filename}]")
        print(f"  {desc}")

        if skip_if_exists(save_path):
            print(f"  ✔ 已存在，跳过")
            done += 1
            continue

        print(f"  → arXiv:{arxiv_id} 下载中...", end="", flush=True)
        if download_arxiv(arxiv_id, save_path):
            kb = os.path.getsize(save_path) / 1024
            print(f" 完成 ({kb:.0f} KB)")
            done += 1
        else:
            print(f" 失败")
            failed.append(("E", filename, f"https://arxiv.org/abs/{arxiv_id}"))
            if os.path.exists(save_path):
                os.remove(save_path)

        time.sleep(4)

    # ── 结果汇总 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  下载完成：{done} / {total} 篇成功")

    if failed:
        print(f"\n  以下 {len(failed)} 篇失败，请手动下载：")
        for cat, name, url in failed:
            print(f"\n  [{cat}] {name}")
            print(f"       {url}")
    else:
        print("  全部下载成功！🎉")

    print("=" * 60)
    print()

    # ── 打印内容说明 ──────────────────────────────────────────────────
    print("各文件夹内容说明：")
    print(f"  D\\ → 工业视觉检测英文论文（写研究现状时引用）")
    print(f"       D1 工业缺陷检测综述  D2 YOLO工业应用综述")
    print(f"       D3 深度学习产品检测  D4 表面缺陷识别综述(IEEE)")
    print()
    print(f"  E\\ → 开题报告技术支撑补充论文")
    print(f"       E1 COCO数据集      → 支撑「采用通用标注格式」的选择")
    print(f"       E2 数据增强策略    → 支撑小样本(500张)的增强方案")
    print(f"       E3 Adam优化器      → 支撑模型训练优化器选择")
    print(f"       E4 批归一化BN      → 支撑模型训练稳定性")
    print(f"       E5 ImageNet数据集  → 支撑迁移学习预训练权重选择")
    print()


if __name__ == "__main__":
    main()
