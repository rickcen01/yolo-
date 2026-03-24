import os
import cv2

# ================= 配置区域 =================
# 原始图片和标签所在的文件夹
SOURCE_DIR = r"d:\biyesheji\gas"
# 输出画好框的图片的文件夹
OUTPUT_DIR = r"d:\biyesheji\gas_visualized"
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取所有图片
files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"开始生成可视化图片，共 {len(files)} 张...")
print(f"源目录: {SOURCE_DIR}")
print(f"输出目录: {OUTPUT_DIR}")

count = 0
for filename in files:
    img_path = os.path.join(SOURCE_DIR, filename)
    label_path = os.path.join(SOURCE_DIR, os.path.splitext(filename)[0] + '.txt')
    
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {filename}")
        continue
    
    height, width, _ = img.shape
    
    # 读取标签 (如果存在)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # 如果文件为空，说明没有标签
        if not lines:
            # 在图片上写个提示
            cv2.putText(img, "No Label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # Normalized xywh
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # 转换为像素坐标
                x1 = int((x_center - w/2) * width)
                y1 = int((y_center - h/2) * height)
                x2 = int((x_center + w/2) * width)
                y2 = int((y_center + h/2) * height)
                
                # 画矩形框 (绿色, 线宽 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 写标签名
                label_text = f"Gas Cylinder ({cls_id})"
                cv2.putText(img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # 没有标签文件
        cv2.putText(img, "No Label File", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 保存可视化图片
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, img)
    count += 1

print(f"\n全部完成！已生成 {count} 张图片。")
print(f"请打开文件夹查看: {OUTPUT_DIR}")
