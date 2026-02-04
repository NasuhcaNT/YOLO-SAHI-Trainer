import cv2
import os
import sys
import numpy as np

# Get imgsz value from GUI (3rd argument), default is 640
TARGET_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 640

# Get project directory and normalize paths
BASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
OUTPUT_DIR = os.path.join(BASE_DIR, f"fixed_{TARGET_SIZE}_dataset")

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

def get_fixed_crop_coords(center_x, center_y, img_w, img_h, size=TARGET_SIZE):
    x1 = max(0, int(center_x - size / 2))
    y1 = max(0, int(center_y - size / 2))
    if x1 + size > img_w:
        x1 = max(0, img_w - size)
    if y1 + size > img_h:
        y1 = max(0, img_h - size)
    return x1, y1, x1 + size, y1 + size

labels = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt") and f != "classes.txt"]

for label_file in labels:
    base_name = os.path.splitext(label_file)[0]
    img_path = None

    for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
        temp_path = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break

    if not img_path:
        continue

    # --- CRITICAL: UTF-8 SAFE IMAGE READ ---
    stream = open(img_path, "rb")
    bytes_data = bytearray(stream.read())
    numpy_array = np.asarray(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    stream.close()

    if img is None:
        print(f"Error: Image could not be read -> {img_path}")
        continue

    h_orig, w_orig, _ = img.shape

    with open(os.path.join(LABEL_DIR, label_file), 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for i, line in enumerate(lines):
        parts = list(map(float, line.split()))
        abs_xc, abs_yc = parts[1] * w_orig, parts[2] * h_orig
        x1, y1, x2, y2 = get_fixed_crop_coords(abs_xc, abs_yc, w_orig, h_orig)

        crop = img[y1:y2, x1:x2]
        new_labels = []

        for l in lines:
            p = list(map(float, l.split()))
            oxc, oyc = p[1] * w_orig, p[2] * h_orig
            if x1 <= oxc <= x2 and y1 <= oyc <= y2:
                nx = (oxc - x1) / TARGET_SIZE
                ny = (oyc - y1) / TARGET_SIZE
                new_labels.append(
                    f"{int(p[0])} {nx:.6f} {ny:.6f} "
                    f"{p[3]*w_orig/TARGET_SIZE:.6f} {p[4]*h_orig/TARGET_SIZE:.6f}"
                )

        save_name = f"{base_name}_tile{i}"

        out_img_path = os.path.join(OUTPUT_DIR, "images", f"{save_name}.jpg")
        is_success, buffer = cv2.imencode(".jpg", crop)
        if is_success:
            with open(out_img_path, "wb") as f:
                f.write(buffer)

        with open(os.path.join(OUTPUT_DIR, "labels", f"{save_name}.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(new_labels))

print("Cropping process completed successfully.")
