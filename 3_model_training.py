import os
import sys
from ultralytics import YOLO

def train_model():
    BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    device = sys.argv[5] if len(sys.argv) > 5 else "0"
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    model_name = sys.argv[7] if len(sys.argv) > 7 else "yolov8n.pt"

    yaml_path = os.path.join(BASE_DIR, "data.yaml")

    model = YOLO(model_name)

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        exist_ok=True,
        project=os.path.join(BASE_DIR, "runs"),
        name="train"
    )

if __name__ == "__main__":
    train_model()
