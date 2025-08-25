#!/usr/bin/env python3
"""
Demo Mode - Test the fruit detection system without mouse control
This is a safe way to test if your model works before enabling auto-play
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import sys
import mss
import pyautogui

# Import configuration
try:
    from config import *
except ImportError:
    print("⚠️ config.py not found, using default settings")
    MODEL_PATH = "trained_models/best_fruit_detection_v11n.pt"
    CONFIDENCE_THRESHOLD = 0.3
    SHOW_DEBUG_WINDOW = True
    VERBOSE_LOGGING = True

class FruitDetectionDemo:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Initialize mss for fast screenshots
        self.sct = mss.mss()
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model."""
        try:
            model_path = Path(MODEL_PATH)
            if not model_path.exists():
                print(f"❌ Model not found at: {model_path}")
                print("Please make sure you have trained a model first.")
                sys.exit(1)
            
            print(f"📥 Loading model: {model_path}")
            self.model = YOLO(str(model_path))
            print("✅ Model loaded successfully!")
            print(f"🎯 Confidence threshold: {self.confidence_threshold}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def detect_fruits(self, frame):
        """Detect fruits in the frame using YOLO."""
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
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
                            'class': cls,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                        })
            
            if VERBOSE_LOGGING and detections:
                print(f"🍎 Detected {len(detections)} fruits")
            
            return detections
            
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"❌ Error in fruit detection: {e}")
            return []
    
    def display_detection_frame(self, frame, detections):
        """Display frame with detection overlays."""
        display_frame = frame.copy()
        
        # Add detection count text
        cv2.putText(display_frame, f"Detections: {len(detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            center_x, center_y = detection['center']
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif conf > 0.4:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence text
            text = f"{conf:.2f}"
            cv2.putText(display_frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(display_frame, (center_x, center_y), 3, color, -1)
            
            # Draw slice path (diagonal line)
            slice_start = (center_x - 20, center_y - 20)
            slice_end = (center_x + 20, center_y + 20)
            cv2.line(display_frame, slice_start, slice_end, color, 2)
        
        return display_frame
    
    def run_demo(self):
        """Run the demo mode."""
        print("🎮 Fruit Detection Demo Mode")
        print("=" * 40)
        print("This demo shows fruit detection without mouse control")
        print("Press 'q' to quit, 's' to save current frame")
        print()
        
        # Get screen dimensions using mss
        monitor = self.sct.monitors[1]  # Primary monitor
        screen_width, screen_height = monitor['width'], monitor['height']
        print(f"🖥️ Screen resolution: {screen_width}x{screen_height}")
        
        # Ask user for demo region
        print("\n🎯 Demo Region Setup:")
        print("1. Open Fruit Ninja or any window with fruits")
        print("2. Position it where you want to test detection")
        print("3. Press Enter when ready")
        
        input("Press Enter to continue...")
        
        print("Move mouse to top-left corner of the test area, then press Enter")
        input("Position mouse at top-left corner and press Enter...")
        
        # Get mouse position using pyautogui (still needed for mouse position)
        top_left = pyautogui.position()
        print(f"Top-left position: {top_left}")
        
        print("Now move mouse to bottom-right corner and press Enter")
        input("Position mouse at bottom-right corner and press Enter...")
        
        bottom_right = pyautogui.position()
        print(f"Bottom-right position: {bottom_right}")
        
        # Calculate region
        x = min(top_left.x, bottom_right.x)
        y = min(top_left.y, bottom_right.y)
        width = abs(bottom_right.x - top_left.x)
        height = abs(bottom_right.y - top_left.y)
        
        demo_region = {'left': x, 'top': y, 'width': width, 'height': height}
        print(f"✅ Demo region set to: {demo_region}")
        
        print("\n🚀 Starting demo... Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # Capture screen region using mss (much faster than pyautogui)
                screenshot = self.sct.grab(demo_region)
                frame = np.array(screenshot)
                
                # Convert from BGRA to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # Detect fruits
                detections = self.detect_fruits(frame)
                
                # Display frame with detections
                display_frame = self.display_detection_frame(frame, detections)
                
                # Show frame
                cv2.imshow("Fruit Detection Demo", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"demo_frame_{timestamp}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"💾 Saved frame as: {filename}")
                
                frame_count += 1
                
                # Show stats every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}, Total detections: {len(detections)}")
                
                # Small delay
                time.sleep(0.03)  # ~30 FPS
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in demo: {e}")
                time.sleep(0.1)
        
        # Cleanup
        cv2.destroyAllWindows()
        print("👋 Demo ended")

def main():
    """Main function."""
    try:
        demo = FruitDetectionDemo()
        demo.run_demo()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
