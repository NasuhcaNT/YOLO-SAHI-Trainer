import os
import yaml
import sys

# Read arguments
BASE_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
TARGET_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 640

# Dataset root directory
DATASET_ROOT = os.path.join(BASE_DIR, f"fixed_{TARGET_SIZE}_dataset")
DATASET_ROOT = DATASET_ROOT.replace("\\", "/")

CLASSES_FILE = os.path.join(BASE_DIR, "labels", "classes.txt")

# Read class names
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
else:
    class_names = ["object"]

yaml_content = {
    'path': DATASET_ROOT,
    'train': 'train/images',
    'val': 'val/images',
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(BASE_DIR, "data.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

print(f"YAML file created: {yaml_path}")
