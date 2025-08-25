#!/usr/bin/env python3
"""
Fruit Detection YOLO Training Script
This script will train a YOLO model on your fruit dataset.
Windows-compatible version.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

def check_dataset():
    """Check if the dataset is properly organized."""
    dataset_path = Path("dataset")
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        print("Please run: python organize_dataset.py <images_dir> <labels_dir>")
        return False
    
    # Check required directories
    required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"❌ Required directory '{dir_name}' not found!")
            return False
    
    # Count files
    train_images = len(list((dataset_path / "train" / "images").glob("*")))
    train_labels = len(list((dataset_path / "train" / "labels").glob("*.txt")))
    val_images = len(list((dataset_path / "val" / "images").glob("*")))
    val_labels = len(list((dataset_path / "val" / "labels").glob("*.txt")))
    
    print(f"✅ Dataset structure verified:")
    print(f"   Training images: {train_images}")
    print(f"   Training labels: {train_labels}")
    print(f"   Validation images: {val_images}")
    print(f"   Validation labels: {val_labels}")
    
    if train_images == 0 or val_images == 0:
        print("❌ No images found in dataset!")
        return False
    
    if train_labels == 0 or val_labels == 0:
        print("❌ No label files found!")
        return False
    
    return True

def update_data_yaml():
    """Update data.yaml with correct class information."""
    # Count unique classes in training labels
    dataset_path = Path("dataset")
    train_labels_dir = dataset_path / "train" / "labels"
    
    class_ids = set()
    for label_file in train_labels_dir.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
        except Exception as e:
            print(f"Warning: Error reading {label_file}: {e}")
    
    if not class_ids:
        print("❌ No valid class IDs found in labels!")
        return False
    
    max_class_id = max(class_ids)
    num_classes = max_class_id + 1
    
    print(f"📊 Found {num_classes} classes with IDs: {sorted(class_ids)}")
    
    # Update data.yaml
    yaml_path = Path("data.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {}
    
    # Preserve existing fruit names if they exist
    existing_names = data.get('names', {})
    
    # Update class information
    data['nc'] = len(class_ids)  # Use actual count, not max+1
    data['names'] = {}
    
    # Map class IDs to names, preserving fruit names
    fruit_names = {
        0: "Banana", 2: "Cherry", 4: "Coconut", 5: "Dragon_Fruit",
        6: "Green_Apple", 7: "Kiwi", 8: "Lemon", 9: "Lime",
        10: "Mango", 11: "Orange", 12: "Passion_Fruit", 13: "Peach",
        14: "Pear", 16: "Pineapple", 18: "Plum", 19: "Pomegranate"
    }
    
    for class_id in sorted(class_ids):
        if class_id in fruit_names:
            data['names'][class_id] = fruit_names[class_id]
        else:
            data['names'][class_id] = f"class_{class_id}"
    
    # Save updated data.yaml
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated data.yaml with {len(class_ids)} classes")
    return True

def get_best_device():
    """Get the best available device for training on Windows."""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def train_model(model_size='n', epochs=100):
    """
    Train the YOLO model.
    
    Args:
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        epochs (int): Number of training epochs
    """
    
    print(f"🚀 Starting training with YOLOv11{model_size} for {epochs} epochs...")
    
    # Force GPU usage for training
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🔧 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ No GPU detected, using CPU (training will be slower)")
    
    print(f"🔧 Using device: {device.upper()}")
    
    # Load model
    model_name = f'yolo11{model_size}.pt'
    print(f"📥 Loading {model_name}...")
    
    try:
        model = YOLO(model_name)
        
        # Verify model is on the correct device
        if device == 'cuda':
            # Force model to GPU
            model.to('cuda')
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✅ Model loaded on CPU")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have internet connection to download the model.")
        return False
    
    # GPU-optimized training configuration for YOLOv11
    if device == 'cuda':
        # Optimize for GPU
        batch_size = 32
        workers = 8
        cache = True
        amp = True
        half = True
        print(f"🚀 GPU training configuration: batch={batch_size}, workers={workers}")
    else:
        # Optimize for CPU
        batch_size = 8
        workers = 0
        cache = False
        amp = False
        half = False
        print(f"🐌 CPU training configuration: batch={batch_size}, workers={workers}")
    
    train_config = {
        'data': str(Path('data.yaml').absolute()),  # Use absolute path
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'name': f'fruit_detection_v11{model_size}',
        'device': device,
        'patience': 20,  # Early stopping
        'save': True,
        'save_period': 10,
        'cache': cache,
        'workers': workers,
        'project': 'runs',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': amp,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save_json': False,
        'save_hybrid': False,
        'save_conf': False,
        'save_crop': False,
        'show': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'conf': 0.001,
        'iou': 0.6,
        'max_det': 300,
        'half': half,
        'dnn': False,
        'source': None,
        'vid_stride': 1,
        'line_thickness': 3,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'classes': None,
        'boxes': True,
    }
    
    # Start training
    try:
        print("🎯 Starting training...")
        results = model.train(**train_config)
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Validate the model
        print("\n🔍 Validating model...")
        metrics = model.val()
        
        print(f"📊 Validation Results:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall: {metrics.box.mr:.3f}")
        
        # Save the trained model with proper Windows path handling
        model_save_dir = Path("trained_models")
        model_save_dir.mkdir(exist_ok=True)
        
        # Save the best model
        best_model_path = model_save_dir / f"best_fruit_detection_v11{model_size}.pt"
        if hasattr(results, 'save_dir') and results.save_dir:
            best_model_source = Path(results.save_dir) / "weights" / "best.pt"
            if best_model_source.exists():
                import shutil
                shutil.copy2(best_model_source, best_model_path)
                print(f"💾 Best model saved to: {best_model_path}")
            else:
                print("⚠️ Best model weights not found, saving current model")
                model.save(best_model_path)
        else:
            model.save(best_model_path)
            print(f"💾 Model saved to: {best_model_path}")
        
        # Export to different formats for Windows compatibility
        try:
            # Export to ONNX for Windows deployment
            onnx_path = model_save_dir / f"best_fruit_detection_v11{model_size}.onnx"
            model.export(format="onnx", imgsz=640)
            print(f"💾 Model exported to ONNX: {onnx_path}")
        except Exception as e:
            print(f"⚠️ ONNX export failed: {e}")
        
        try:
            # Export to TorchScript for Windows deployment
            torchscript_path = model_save_dir / f"best_fruit_detection_v11{model_size}.torchscript"
            model.export(format="torchscript", imgsz=640)
            print(f"💾 Model exported to TorchScript: {torchscript_path}")
        except Exception as e:
            print(f"⚠️ TorchScript export failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training workflow."""
    print("🍎 Fruit Detection YOLO Training (Windows)")
    print("=" * 50)
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    # Update data.yaml
    if not update_data_yaml():
        sys.exit(1)
    
    # Get training parameters
    print("\n📋 Training Configuration:")
    model_size = input("Model size (n/s/m/l/x) [n]: ").strip() or 'n'
    epochs = input("Number of epochs [100]: ").strip() or '100'
    
    try:
        epochs = int(epochs)
    except ValueError:
        print("Invalid epochs, using default 100")
        epochs = 100
    
    # Validate model size
    if model_size not in ['n', 's', 'm', 'l', 'x']:
        print("Invalid model size, using 'n' (nano)")
        model_size = 'n'
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Model: YOLOv11{model_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Dataset: {Path('dataset').absolute()}")
    
    # Confirm training
    confirm = input("\nProceed with training? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)
    
    # Start training
    success = train_model(model_size, epochs)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("You can now use your trained model for inference.")
        print("\nNext steps:")
        print("1. Update fruits.py with the path to your trained model")
        print("2. Test the model on new images")
        print("3. Fine-tune if needed")
        print(f"\n📁 Models saved in: {Path('trained_models').absolute()}")
    else:
        print("\n❌ Training failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
