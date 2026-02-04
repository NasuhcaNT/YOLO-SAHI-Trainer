import os
import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QTextEdit, QFrame
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt


# Thread class to run pipeline steps without freezing the UI
class ProcessThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        
        root = self.config['root']
        try:
            self.log_signal.emit(f"🚀 Pipeline started: {root}")

            # 1. CROP
            self.log_signal.emit("📦 1/5: Cropping data...")
            subprocess.run(
                [sys.executable, "1_crop_data.py", root, self.config['imgsz']],
                check=True
            )

            # 2. YAML
            self.log_signal.emit("📝 2/5: Creating data.yaml...")
            subprocess.run(
                [sys.executable, "1_5_create_yaml.py", root, self.config['imgsz']],
                check=True
            )

            # 3. SPLIT
            self.log_signal.emit("✂️ 3/5: Splitting dataset...")
            subprocess.run(
                [sys.executable, "2_split_data.py", root, self.config['imgsz'], self.config['ratio']],
                check=True
            )

            # 4. TRAIN
            self.log_signal.emit("🔥 4/5: Training YOLO model...")
            subprocess.run([
                sys.executable,
                "3_model_training.py",
                root,
                str(self.config['epochs']),
                str(self.config['imgsz']),
                str(self.config['batch']),
                str(self.config['device']),
                str(self.config['workers']),
                self.config['model_name']
            ], check=True)

            # 5. SAHI
            self.log_signal.emit("🔍 5/5: Running SAHI inference...")
            subprocess.run(
                [sys.executable, "4_sahi_inference.py", root, self.config['imgsz']],
                check=True
            )

            self.log_signal.emit("✅ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")

        except Exception as e:
            self.log_signal.emit(f"❌ ERROR OCCURRED: {str(e)}")

        self.finished_signal.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO & SAHI Pipeline Manager")
        self.setMinimumSize(600, 700)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("YOLO Training & SAHI Inference Panel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(
            "Select project folder (must contain images and labels directories)"
        )
        self.btn_browse = QPushButton("Browse Folder")
        self.btn_browse.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.path_input)
        folder_layout.addWidget(self.btn_browse)
        layout.addLayout(folder_layout)

        # Settings frame
        self.settings_frame = QFrame()
        self.settings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        settings_layout = QVBoxLayout(self.settings_frame)

        self.inputs = {}
        params = [
            ("Model Name:", "model_name", "yolov8n.pt"),
            ("Epochs:", "epochs", "2"),
            ("Batch Size:", "batch", "8"),
            ("Image Size:", "imgsz", "640"),
            ("Train Ratio:", "ratio", "0.8"),
            ("Device (GPU:0 / cpu):", "device", "0"),
            ("Workers:", "workers", "0"),
        ]

        for label_text, key, default in params:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            edit = QLineEdit(default)
            row.addWidget(edit)
            self.inputs[key] = edit
            settings_layout.addLayout(row)

        layout.addWidget(self.settings_frame)

        # Run button
        self.btn_run = QPushButton("START PIPELINE")
        self.btn_run.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; height: 40px;"
        )
        self.btn_run.clicked.connect(self.start_pipeline)
        layout.addWidget(self.btn_run)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1; font-family: Consolas;"
        )
        layout.addWidget(self.log_output)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.path_input.setText(folder)

    def start_pipeline(self):
        if not self.path_input.text():
            self.log_output.append("⚠️ Please select a project folder first!")
            return

        config = {
            'root': self.path_input.text(),
            'model_name': self.inputs['model_name'].text(),
            'epochs': self.inputs['epochs'].text(),
            'batch': self.inputs['batch'].text(),
            'imgsz': self.inputs['imgsz'].text(),
            'ratio': self.inputs['ratio'].text(),
            'device': self.inputs['device'].text(),
            'workers': self.inputs['workers'].text(),
        }

        self.btn_run.setEnabled(False)
        self.thread = ProcessThread(config)
        self.thread.log_signal.connect(self.log_output.append)
        self.thread.finished_signal.connect(
            lambda: self.btn_run.setEnabled(True)
        )
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
