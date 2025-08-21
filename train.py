from ultralytics import YOLO

# Load a YOLO model (e.g., YOLOv8n - nano version)
model = YOLO('yolo11m.pt')  # Replace with the desired YOLO model checkpoint

# Train the model
model.train(
    data='path/to/dataset.yaml',  # Path to the dataset YAML file
    epochs=50,                   # Number of training epochs
    batch=16,                    # Batch size
    imgsz=640,                   # Image size
    name='fruit_detection',      # Name of the training run
    device=0                     # GPU device (use 'cpu' for CPU training)
)