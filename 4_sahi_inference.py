import os
import sys
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640

def run_sahi():
    BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        os.path.join(BASE_DIR, "runs", "train", "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(BASE_DIR, "best.pt")
    ]

    MODEL_PATH = None
    for path in possible_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"Model found: {MODEL_PATH}")
            break

    if MODEL_PATH is None:
        print("ERROR: best.pt not found!")
        return

    test_image_dir = os.path.join(BASE_DIR, "images")
    try:
        test_images = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.jpg', '.png'))]
        image_path = os.path.join(test_image_dir, test_images[0])
    except:
        print("ERROR: No test image found!")
        return

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=0.3,
        device="cuda:0",
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=imgsz,
        slice_width=imgsz,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    output_dir = os.path.join(BASE_DIR, "sahi_results")
    os.makedirs(output_dir, exist_ok=True)
    result.export_visuals(export_dir=output_dir)

    print(f"Inference completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    run_sahi()
