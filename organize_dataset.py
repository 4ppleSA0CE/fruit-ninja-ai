import os
import shutil
import random
from pathlib import Path
import glob

def organize_dataset(images_dir, labels_dir, output_dir="dataset", train_split=0.8):
    """
    Organize existing images and labels into YOLO dataset structure.
    
    Args:
        images_dir (str): Directory containing your image files
        labels_dir (str): Directory containing your .txt label files
        output_dir (str): Output directory for the organized dataset
        train_split (float): Fraction of data to use for training (0.8 = 80% train, 20% val)
    """
    
    # Create output directory structure
    output_path = Path(output_dir)
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} image files in {images_dir}")
    
    # Find all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"Found {len(label_files)} label files in {labels_dir}")
    
    # Create a mapping of image names to label names
    image_to_label = {}
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_name_without_ext = os.path.splitext(img_name)[0]
        
        # Look for corresponding label file
        label_name = f"{img_name_without_ext}.txt"
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            image_to_label[img_path] = label_path
        else:
            print(f"Warning: No label file found for {img_name}")
    
    print(f"Found {len(image_to_label)} image-label pairs")
    
    # Convert to list and shuffle
    paired_files = list(image_to_label.items())
    random.shuffle(paired_files)
    
    # Split into train/val
    split_idx = int(len(paired_files) * train_split)
    train_pairs = paired_files[:split_idx]
    val_pairs = paired_files[split_idx:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # Copy training files
    print("Copying training files...")
    for img_path, label_path in train_pairs:
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Copy image
        shutil.copy2(img_path, train_images / img_name)
        
        # Copy label
        shutil.copy2(label_path, train_labels / label_name)
    
    # Copy validation files
    print("Copying validation files...")
    for img_path, label_path in val_pairs:
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Copy image
        shutil.copy2(img_path, val_images / img_name)
        
        # Copy label
        shutil.copy2(label_path, val_labels / label_name)
    
    print(f"\nDataset organized successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Training images: {len(train_pairs)}")
    print(f"Validation images: {len(val_pairs)}")
    
    # Verify the organized dataset
    verify_dataset(output_dir)

def verify_dataset(dataset_dir="dataset"):
    """Verify the dataset structure and files."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return False
    
    # Check structure
    required_dirs = [
        "train/images", "train/labels",
        "val/images", "val/labels"
    ]
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"Error: Required directory '{dir_name}' not found!")
            return False
    
    # Count files
    train_images = len(list((dataset_path / "train" / "images").glob("*")))
    train_labels = len(list((dataset_path / "train" / "labels").glob("*.txt")))
    val_images = len(list((dataset_path / "val" / "images").glob("*")))
    val_labels = len(list((dataset_path / "val" / "labels").glob("*.txt")))
    
    print(f"\nDataset verification:")
    print(f"  Training images: {train_images}")
    print(f"  Training labels: {train_labels}")
    print(f"  Validation images: {val_images}")
    print(f"  Validation labels: {val_labels}")
    
    if train_images == 0 or val_images == 0:
        print("Error: No images found in dataset!")
        return False
    
    if train_labels == 0 or val_labels == 0:
        print("Warning: No label files found!")
        return False
    
    print("Dataset structure is valid!")
    return True

def check_label_format(labels_dir):
    """Check the format of your label files."""
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print("No label files found!")
        return
    
    print(f"Checking format of {len(label_files)} label files...")
    
    # Check first few label files
    for i, label_file in enumerate(label_files[:5]):  # Check first 5 files
        print(f"\nChecking {os.path.basename(label_file)}:")
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                print(f"  Lines: {len(lines)}")
                
                for j, line in enumerate(lines[:3]):  # Show first 3 annotations
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        print(f"    Line {j+1}: Class {class_id}, Center ({x_center:.3f}, {y_center:.3f}), Size ({width:.3f}, {height:.3f})")
                        
                        # Validate YOLO format
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            print(f"      Warning: Values should be normalized between 0 and 1")
                    else:
                        print(f"    Line {j+1}: Invalid format - {line.strip()}")
                        
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    if len(label_files) > 5:
        print(f"\n... and {len(label_files) - 5} more files")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python organize_dataset.py <images_directory> <labels_directory>")
        print("Example: python organize_dataset.py ./images ./labels")
        print("\nThis will organize your existing images and labels into the proper YOLO dataset structure.")
        sys.exit(1)
    
    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory '{images_dir}' not found!")
        sys.exit(1)
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory '{labels_dir}' not found!")
        sys.exit(1)
    
    # Check label format first
    print("Checking label format...")
    check_label_format(labels_dir)
    
    # Organize dataset
    print(f"\nOrganizing dataset...")
    organize_dataset(images_dir, labels_dir)
