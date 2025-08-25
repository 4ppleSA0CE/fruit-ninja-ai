# 🍎 Fruit Ninja AI - YOLO Training Project

A complete YOLO training pipeline for fruit detection using annotated screenshots from Fruit Ninja game.

## 🎯 Project Overview

This project trains a YOLO (You Only Look Once) object detection model to identify various fruits in Fruit Ninja game screenshots. The model can detect 16 different fruit types with high accuracy.

## 🍓 Detected Fruits

The model is trained to detect the following 16 fruit types:
- **Banana** - Class 0
- **Cherry** - Class 1  
- **Coconut** - Class 2
- **Dragon Fruit** - Class 3
- **Green Apple** - Class 4
- **Kiwi** - Class 5
- **Lemon** - Class 6
- **Lime** - Class 7
- **Mango** - Class 8
- **Orange** - Class 9
- **Passion Fruit** - Class 10
- **Peach** - Class 11
- **Pear** - Class 12
- **Pineapple** - Class 13
- **Plum** - Class 14
- **Pomegranate** - Class 15

## 📁 Project Structure

```
fruit-ninja-ai/
├── dataset/                 # Organized training dataset
│   ├── train/              # Training images and labels
│   │   ├── images/         # Training images
│   │   └── labels/         # Training labels (.txt)
│   └── val/                # Validation images and labels
│       ├── images/         # Validation images
│       └── labels/         # Validation labels (.txt)
├── screenshots/             # Original game screenshots
├── labels/                  # Original YOLO format labels
├── train_fruits.py          # Main training script
├── organize_dataset.py      # Dataset organization utility
├── fruits.py                # Inference script for game automation
├── data.yaml                # Dataset configuration
├── requirements.txt         # Python dependencies
└── TRAINING_GUIDE.md        # Comprehensive training guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize Dataset
```bash
python3 organize_dataset.py ./screenshots ./labels
```

### 3. Train Model
```bash
python3 train_fruits.py
```

### 4. Use Trained Model
```bash
python3 fruits.py
```

## 🔧 Training Configuration

- **Model**: YOLOv11n (nano version)
- **Device**: Mac GPU (MPS) or CPU
- **Image Size**: 640x640 pixels
- **Batch Size**: 8 (optimized for Mac MPS)
- **Epochs**: 100 (configurable)
- **Classes**: 16 fruit types

## 📊 Dataset Statistics

- **Total Images**: 816 screenshots
- **Total Labels**: 643 annotated objects
- **Training Set**: 514 images (80%)
- **Validation Set**: 129 images (20%)
- **Class Distribution**: Balanced across all fruit types

## 🎮 Game Automation

The `fruits.py` script provides real-time fruit detection and can be used for:
- Automated fruit slicing
- Score optimization
- Game analysis
- Training data collection

## 🛠️ Technical Details

- **Framework**: Ultralytics YOLO
- **Python**: 3.8+
- **Dependencies**: PyTorch, OpenCV, NumPy, PyYAML
- **Platform**: macOS (MPS GPU support), Linux, Windows

## 🙏 Acknowledgments

- Ultralytics for the YOLO framework
- Fruit Ninja game developers
- Open source computer vision community

---

**Happy Training! 🚀🍎**
