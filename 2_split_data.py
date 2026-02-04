import os
import random
import shutil
import sys

def split_data():
    BASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    TARGET_SIZE = sys.argv[2] if len(sys.argv) > 2 else "640"

    raw_ratio = sys.argv[3] if len(sys.argv) > 3 else "0.8"

    try:
        TRAIN_RATIO = float(raw_ratio)
        if TRAIN_RATIO > 1:
            TRAIN_RATIO /= 100.0
    except:
        TRAIN_RATIO = 0.8

    print("--- DATA SPLITTING STARTED ---")
    print(f"Train ratio: {TRAIN_RATIO}")

    DATA_DIR = os.path.join(BASE_DIR, f"fixed_{TARGET_SIZE}_dataset")
    img_source = os.path.join(DATA_DIR, "images")
    lbl_source = os.path.join(DATA_DIR, "labels")

    if not os.path.exists(img_source):
        print(f"ERROR: {img_source} not found!")
        return

    all_images = [f for f in os.listdir(img_source) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(all_images)

    total = len(all_images)
    split_idx = int(total * TRAIN_RATIO)

    if split_idx >= total and total > 1:
        split_idx = int(total * 0.8)

    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    print(f"Total images: {total}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    for s in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(DATA_DIR, s, sub), exist_ok=True)

    for f in train_files:
        shutil.move(os.path.join(img_source, f), os.path.join(DATA_DIR, "train", "images", f))
        lbl = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(os.path.join(lbl_source, lbl)):
            shutil.move(os.path.join(lbl_source, lbl), os.path.join(DATA_DIR, "train", "labels", lbl))

    for f in val_files:
        shutil.move(os.path.join(img_source, f), os.path.join(DATA_DIR, "val", "images", f))
        lbl = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(os.path.join(lbl_source, lbl)):
            shutil.move(os.path.join(lbl_source, lbl), os.path.join(DATA_DIR, "val", "labels", lbl))

    print("✅ Data split completed successfully.")

if __name__ == "__main__":
    split_data()
