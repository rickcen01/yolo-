import os
import shutil
import random

# Define paths
source_dir = r"d:\biyesheji\gas"  # Where your images and labels (txt) are
dataset_dir = r"d:\biyesheji\gas_dataset"

# Create directories
images_train = os.path.join(dataset_dir, 'images', 'train')
images_val = os.path.join(dataset_dir, 'images', 'val')
labels_train = os.path.join(dataset_dir, 'labels', 'train')
labels_val = os.path.join(dataset_dir, 'labels', 'val')

os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# Get all images
files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(files)

# Split 80/20
split_idx = int(len(files) * 0.8)
train_files = files[:split_idx]
val_files = files[split_idx:]

def copy_files(file_list, src, dest_img, dest_lbl):
    for filename in file_list:
        # Copy image
        shutil.copy2(os.path.join(src, filename), os.path.join(dest_img, filename))
        
        # Copy label if exists
        label_name = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(src, label_name)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(dest_lbl, label_name))
        else:
            print(f"Warning: No label found for {filename}")

print("Copying train files...")
copy_files(train_files, source_dir, images_train, labels_train)
print("Copying val files...")
copy_files(val_files, source_dir, images_val, labels_val)

print("Done! Dataset is ready in d:\\biyesheji\\gas_dataset")
