# YOLO-SAHI-Trainer
An end-to-end YOLO training pipeline with automated annotation-aware cropping, dataset preparation, and integrated SAHI inference for robust small object detection in high-resolution images.

Here is a professional, high-quality README.md file written in English, specifically tailored to the technical summary you provided.

🚀 NanoDetect Suite
An end-to-end YOLOv8 training pipeline with automated annotation-aware cropping, dataset preparation, and integrated SAHI inference for robust small object detection in high-resolution images.

📌 Overview
Detecting small objects in high-resolution imagery (e.g., drone footage, 4K security feeds, medical imaging) is a major challenge for standard object detection models. When high-res images are downscaled to fit YOLO inputs, critical pixel information for small objects is lost.

NanoDetect Suite overcomes this by providing a professional-grade GUI to manage an automated pipeline that intelligently slices high-resolution data while preserving annotations, training the model, and utilizing SAHI (Slicing Aided Hyper Inference) for maximum detection accuracy.

✨ Key Features
🎯 Annotation-Aware Cropping: Intelligently slices large images into smaller tiles while ensuring labels are correctly mapped and small objects are preserved at their original scale.

🛠 Automated Dataset Management: Generates data.yaml dynamically and handles the distribution of files into 80% Train / 20% Val splits automatically.

📊 Real-time Training Monitor: A modern PyQt6 interface to monitor training progress, epoch counts, and dataset statistics in real-time.

🔍 Integrated SAHI Inference: Built-in support for Slicing Aided Hyper Inference to ensure robust detection performance on massive images after training.

💻 Flexible Device Support: Seamlessly switch between NVIDIA GPU (0, 1, ...) or cpu for training and inference.

🛠 Installation
Clone the Repository:

Bash
git clone https://github.com/yourusername/YOLO-SAHI-Trainer.git
cd NanoDetect-Suite
Install Dependencies:

Bash
pip install -r requirements.txt
(Requirements: ultralytics, sahi, PyQt6, pyyaml, opencv-python)

🚀 Getting Started
Launch the management console:

Bash
python main_gui.py
Workflow Steps:
Select Workspace: Choose the root directory containing your raw images and labels.

Configure Parameters: Set your target imgsz (e.g., 640), epochs, and batch size.

Select Device: Type 0 for the first NVIDIA GPU or cpu for processor-based training.

Execute: Click the green "START PIPELINE" button. The system will handle everything from cropping to final inference.

📂 Repository Structure
Plaintext
NanoDetect-Suite/

├── main_gui.py          # Central GUI management console

├── 1_crop_data.py       # Smart annotation-aware slicing logic

├── 1.5_create_yaml.py   # Dynamic YOLO configuration generator

├── 2_split_data.py      # Automated dataset distributor (Train/Val)

├── 3_Model_Train.py     # YOLOv8 core training engine

└── 4_SAHI_Inference.py  # Advanced inference script for small objects

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
