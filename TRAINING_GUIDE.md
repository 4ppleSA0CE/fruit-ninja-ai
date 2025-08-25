# 🍎 Fruit Detection YOLO Training Guide

This guide will walk you through training a YOLO model for fruit detection using your annotated dataset.

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Your annotated dataset** with images and YOLO format labels
3. **Internet connection** (to download YOLO models)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Organize Your Dataset

If you have a zip file:
```bash
python prepare_dataset.py your_annotated_images.zip
```

If you have separate images and labels folders:
```bash
python organize_dataset.py ./images ./labels
```

This will create the proper YOLO dataset structure:
```
dataset/
├── train/
│   ├── images/     # Training images
│   └── labels/     # Training labels (.txt files)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels (.txt files)
└── data.yaml       # Dataset configuration
```

### Step 3: Train Your Model

```bash
python train_fruits.py
```

The script will:
- ✅ Verify your dataset structure
- 📊 Automatically detect the number of classes
- 🎯 Guide you through training configuration
- 🚀 Start training with optimal parameters
- 📈 Show training progress and results

## 📁 Dataset Structure Requirements

Your dataset must follow this structure:

```
dataset/
├── train/
│   ├── images/     # Training images (.jpg, .png, etc.)
│   └── labels/     # Training labels (.txt files)
├── val/
│   ├── images/     # Validation images
│   └── labels/     # Validation labels
└── data.yaml       # Dataset configuration
```

## 🏷️ Label Format

Your `.txt` label files must be in YOLO format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer starting from 0
- `x_center, y_center`: Normalized center coordinates (0-1)
- `width, height`: Normalized dimensions (0-1)

Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.3
```

## 🎯 Training Configuration

### Model Sizes
- **YOLOv8n** (nano): Fastest, smallest, good for testing
- **YOLOv8s** (small): Good balance of speed and accuracy
- **YOLOv8m** (medium): Better accuracy, slower
- **YOLOv8l** (large): High accuracy, slower
- **YOLOv8x** (extra large): Best accuracy, slowest

### Training Parameters
- **Epochs**: 100 (default), increase for better accuracy
- **Batch Size**: 16 (reduce if you run out of memory)
- **Image Size**: 640x640 pixels
- **Learning Rate**: Automatically optimized

## 📊 Training Process

1. **Dataset Validation**: Checks structure and counts files
2. **Class Detection**: Automatically counts unique classes
3. **Model Loading**: Downloads pre-trained YOLO model
4. **Training**: Runs for specified number of epochs
5. **Validation**: Evaluates model performance
6. **Model Export**: Saves trained model

## 📈 Monitoring Training

Training progress is saved in `runs/train/fruit_detection_v8*/`:
- **Training curves** (loss, mAP)
- **Validation results**
- **Best model weights**
- **Training logs**

## 🔍 Using Your Trained Model

After training, update `fruits.py`:

```python
# Load the trained YOLO model
model = YOLO("runs/train/fruit_detection_v8n/weights/best.pt")
```

## 🛠️ Troubleshooting

### Common Issues

1. **"Dataset directory not found"**
   - Run the organization script first
   - Check file paths

2. **"No label files found"**
   - Ensure labels are in `.txt` format
   - Check label file names match image names

3. **"Out of memory"**
   - Reduce batch size (e.g., from 16 to 8)
   - Use smaller model size (e.g., 'n' instead of 'm')

4. **"Model download failed"**
   - Check internet connection
   - Try again later

### Performance Tips

1. **Use GPU** if available (automatically detected)
2. **Start with nano model** for quick testing
3. **Increase epochs** for better accuracy
4. **Use data augmentation** (enabled by default)

## 📚 Advanced Configuration

### Custom Training Parameters

Edit `train_fruits.py` to modify:
- Learning rate schedules
- Data augmentation
- Loss weights
- Early stopping criteria

### Multi-Class Detection

If you have multiple fruit types:
1. Update class names in `data.yaml`
2. Ensure labels use correct class IDs
3. Set `single_cls=False` in training config

## 🎉 Next Steps

After successful training:

1. **Test on new images**:
   ```python
   results = model("test_image.jpg")
   ```

2. **Real-time detection**:
   - Use `fruits.py` for game automation
   - Adjust confidence thresholds
   - Fine-tune detection parameters

3. **Model optimization**:
   - Export to different formats
   - Quantize for faster inference
   - Test on edge devices

## 📞 Support

If you encounter issues:
1. Check the error messages
2. Verify dataset structure
3. Ensure all dependencies are installed
4. Check file permissions

## 🔗 Useful Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Label Format](https://roboflow.com/formats/yolo-format)
- [Dataset Preparation Best Practices](https://docs.ultralytics.com/guides/datasets/)

---

**Happy Training! 🚀🍎**
