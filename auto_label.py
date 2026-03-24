from ultralytics import YOLO
import os

# ================= 配置区域 =================
# 图片所在文件夹
IMAGE_DIR = r"d:\biyesheji\gas"
# 使用最强的 YOLO 模型
MODEL_NAME = 'yolo11x.pt' 

# 根据诊断结果，钢瓶（上半身）经常被误识别为以下物体：
# 39: bottle (瓶子)
# 71: sink (水槽 - 钢瓶阀门部分很像水槽)
# 41: cup (杯子)
# 56: chair (椅子 - 某些角度的支架部分)
# 10: fire hydrant (消防栓)
# 75: vase (花瓶)
SOURCE_CLASS_IDS = [39, 71, 41, 56, 10, 75]

# 最终保存的类别 ID (0: gas_cylinder)
TARGET_CLASS_ID = 0

# 置信度阈值 (设为 0.05 以捕获 Pic_gas_100.png 中仅 0.07 的目标)
CONF_THRESHOLD = 0.05
# ===========================================

def auto_label_final():
    print(f"加载模型 {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(f"开始处理文件夹: {IMAGE_DIR}")
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    count = 0
    labeled_count = 0
    
    for file in files:
        image_path = os.path.join(IMAGE_DIR, file)
        txt_path = os.path.join(IMAGE_DIR, os.path.splitext(file)[0] + ".txt")

        # 注意：这里去掉了“如果文件存在则跳过”的逻辑，
        # 因为我们更新了策略，想要覆盖之前可能漏检的结果。
        # 如果你已经手工标了一些，请先备份！
        # 为了安全起见，我们还是保留跳过逻辑，但如果文件为空或者只有0行，可以覆盖。
        if os.path.exists(txt_path):
             with open(txt_path, 'r') as f:
                 if len(f.readlines()) > 0:
                     print(f"跳过 (已存在有效标签): {file}")
                     continue

        # 推理
        results = model(image_path, device='cpu', conf=CONF_THRESHOLD, verbose=False)

        detected_boxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 如果是我们在找的任何一个相似类别
                if cls_id in SOURCE_CLASS_IDS:
                    # 获取归一化坐标
                    x, y, w, h = box.xywhn[0].tolist()
                    detected_name = result.names[cls_id]
                    
                    # 简单的过滤：钢瓶通常是竖长的 (h > w)，如果检测到特别扁的(比如误检的横向物体)，可以过滤掉
                    # 但考虑到是“上半身”，有时可能接近正方形，暂不过滤，留给人工筛选
                    
                    detected_boxes.append(f"{TARGET_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    # print(f"  -> {file}: 识别到 {detected_name} ({conf:.2f})")

        # 写入文件
        if detected_boxes:
            with open(txt_path, 'w') as f:
                f.write('\n'.join(detected_boxes))
            print(f"自动标注完成: {file} -> 找到了 {len(detected_boxes)} 个目标")
            labeled_count += 1
        else:
            print(f"未检测到目标: {file} (阈值已降至 {CONF_THRESHOLD})")
        
        count += 1

    print(f"\n全部完成！")
    print(f"共扫描 {count} 张图片，成功标注 {labeled_count} 张。")
    print("请务必使用 labelImg 打开检查！")
    print("提示：由于阈值很低，可能会有不少误检（杂物），请在 labelImg 中按 'd' 快速切换并删除多余框。")

if __name__ == '__main__':
    auto_label_final()
