#!/usr/bin/env python3
"""
Detailed test script to see all model detections regardless of confidence
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

def test_model_detailed():
    """Test the trained model and show ALL detections."""
    print("🔍 Detailed YOLO Model Test")
    print("=" * 40)
    
    # Load the trained model
    try:
        model_path = Path("trained_models/best_fruit_detection_v11n.pt")
        if not model_path.exists():
            print(f"❌ Model not found at: {model_path}")
            return False
        
        print(f"📥 Loading model: {model_path}")
        model = YOLO(str(model_path))
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test on multiple sample images
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists():
        print("❌ Screenshots directory not found!")
        return False
    
    # Find sample images
    sample_images = list(screenshots_dir.glob("*.png"))[:5]  # Test first 5 images
    if not sample_images:
        print("❌ No sample images found in screenshots folder!")
        return False
    
    print(f"🖼️ Testing on {len(sample_images)} sample images...")
    
    for i, test_image_path in enumerate(sample_images):
        print(f"\n--- Image {i+1}: {test_image_path.name} ---")
        
        try:
            # Load the image
            image = cv2.imread(str(test_image_path))
            if image is None:
                print("❌ Could not load the image!")
                continue
            
            print(f"📏 Dimensions: {image.shape}")
            
            # Run inference with very low confidence threshold
            print("🔍 Running inference...")
            results = model(image, verbose=False, conf=0.01)  # Very low confidence threshold
            
            # Process results
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': cls
                        })
            
            print(f"🎯 Found {len(detections)} detections:")
            
            if detections:
                # Sort by confidence
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                for j, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    cls = detection['class']
                    print(f"  {j+1}. Class {cls}, Confidence: {conf:.4f}, Box: ({x1}, {y1}, {x2}, {y2})")
                
                # Show the image with detections
                display_image = image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    conf = detection['confidence']
                    
                    # Color based on confidence
                    if conf > 0.7:
                        color = (0, 255, 0)  # Green for high confidence
                    elif conf > 0.4:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Draw bounding box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw confidence text
                    text = f"{conf:.3f}"
                    cv2.putText(display_image, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Resize image if too large for display
                height, width = display_image.shape[:2]
                if width > 1200 or height > 800:
                    scale = min(1200/width, 800/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))
                
                # Show the image
                window_name = f"Detections - {test_image_path.name}"
                cv2.imshow(window_name, display_image)
                print(f"🖼️ Press any key to continue to next image...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
                
            else:
                print("⚠️ No detections found even with low confidence threshold")
                print("This suggests the model may need more training or the images don't contain fruits")
        
        except Exception as e:
            print(f"❌ Error testing image {test_image_path}: {e}")
            continue
    
    cv2.destroyAllWindows()
    return True

def main():
    """Main function."""
    success = test_model_detailed()
    
    if success:
        print("\n✅ Detailed test completed!")
        print("\n📊 Analysis:")
        print("- If you see many low-confidence detections, lower the confidence threshold")
        print("- If you see no detections, the model may need more training")
        print("- If you see good detections, you can proceed with auto-play")
        print("\nNext steps:")
        print("1. Adjust confidence threshold in config.py if needed")
        print("2. Run auto-play with: python3 fruits.py")
    else:
        print("\n❌ Detailed test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
