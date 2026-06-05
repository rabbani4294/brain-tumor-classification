import os
import shutil
import random

SOURCE_DIR = r"D:\Brain_tumor_project\dataset\training"
OUTPUT_DIR = r"D:\Brain_tumor_project\dataset_split"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

classes = os.listdir(SOURCE_DIR)

for split in SPLIT_RATIO:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

for cls in classes:

    class_dir = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_dir)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * SPLIT_RATIO["train"])
    val_end = train_end + int(total * SPLIT_RATIO["val"])

    for i, img in enumerate(images):

        src_path = os.path.join(class_dir, img)

        if i < train_end:
            dst_path = os.path.join(OUTPUT_DIR, "train", cls, img)

        elif i < val_end:
            dst_path = os.path.join(OUTPUT_DIR, "val", cls, img)

        else:
            dst_path = os.path.join(OUTPUT_DIR, "test", cls, img)

        shutil.copy(src_path, dst_path)

print("Dataset split completed successfully!")