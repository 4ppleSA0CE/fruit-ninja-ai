#!/usr/bin/env python3
"""
Fruit Ninja AI Auto-Player
Uses trained YOLO model to detect fruits and automatically slice them
"""

import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from pathlib import Path
import sys

# Import configuration
try:
    from config import *
except ImportError:
    print("⚠️ config.py not found, using default settings")
    MODEL_PATH = "trained_models/best_fruit_detection_v11n.pt"
    CONFIDENCE_THRESHOLD = 0.3
    MOUSE_MOVE_DURATION = 0.05
    MOUSE_DRAG_DURATION = 0.1
    SLICE_DISTANCE = 20
    MIN_DETECTION_INTERVAL = 0.1
    FRAME_DISPLAY_INTERVAL = 30
    SCREEN_CAPTURE_DELAY = 0.01
    SHOW_DEBUG_WINDOW = False
    VERBOSE_LOGGING = True

pyautogui.PAUSE = MOUSE_PAUSE if 'MOUSE_PAUSE' in locals() else 0.01

class SimpleToggleManager:
    def __init__(self):
        self.is_active = False
        self.input_thread = None
        self.quit_requested = False
        
    def start_monitoring(self):
        self.input_thread = threading.Thread(target=self._monitor_input, daemon=True)
        self.input_thread.start()
        print("⌨️ Input monitoring started: Press 't' + Enter to toggle auto-play")
    
    def _monitor_input(self):
        while not self.quit_requested:
            try:
                key = input("Press 't' + Enter to toggle auto-play, 'q' + Enter to quit: ").strip().lower()
                if key == 't':
                    self.is_active = not self.is_active
                    status = "ACTIVATED" if self.is_active else "DEACTIVATED"
                    print(f"🎯 Auto-play {status}!")
                elif key == 'q':
                    print("🛑 Quit command received")
                    self.quit_requested = True
                    break
            except (EOFError, KeyboardInterrupt):
                break
        
        print("🔄 Input monitoring stopped")
    
    def request_quit(self):
        self.quit_requested = True

class FruitNinjaAI:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.game_region = None
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.last_detection_time = 0
        self.min_detection_interval = MIN_DETECTION_INTERVAL
        
        self.toggle_manager = SimpleToggleManager()
        
        self.load_model()
        self.setup_mouse_control()
    
    def load_model(self):
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
    
    def setup_mouse_control(self):
        screen_width, screen_height = pyautogui.size()
        print(f"🖥️ Screen resolution: {screen_width}x{screen_height}")
        
        game_width = DEFAULT_GAME_WIDTH if 'DEFAULT_GAME_WIDTH' in locals() else 800
        game_height = DEFAULT_GAME_HEIGHT if 'DEFAULT_GAME_HEIGHT' in locals() else 600
        game_x = (screen_width - game_width) // 2
        game_y = (screen_height - game_height) // 2
        
        self.game_region = (game_x, game_y, game_width, game_height)
        print(f"🎮 Default game region: {self.game_region}")
        
        self.adjust_game_region()
    
    def adjust_game_region(self):
        print("\n🎯 Game Region Setup:")
        print("1. Open Fruit Ninja in a window")
        print("2. Position it where you want to play")
        print("3. Press Enter when ready")
        
        input("Press Enter to continue...")
        
        print("Move mouse to top-left corner of game window, then press Enter")
        input("Position mouse at top-left corner and press Enter...")
        
        top_left = pyautogui.position()
        print(f"Top-left position: {top_left}")
        
        print("Now move mouse to bottom-right corner and press Enter")
        input("Position mouse at bottom-right corner and press Enter...")
        
        bottom_right = pyautogui.position()
        print(f"Bottom-right position: {bottom_right}")
        
        x = min(top_left.x, bottom_right.x)
        y = min(top_left.y, bottom_right.y)
        width = abs(bottom_right.x - top_left.x)
        height = abs(bottom_right.y - top_left.y)
        
        self.game_region = (x, y, width, height)
        print(f"✅ Game region set to: {self.game_region}")
        
        confirm = input("Use this region? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("Using default region")
            screen_width, screen_height = pyautogui.size()
            game_width = DEFAULT_GAME_WIDTH if 'DEFAULT_GAME_WIDTH' in locals() else 800
            game_height = DEFAULT_GAME_HEIGHT if 'DEFAULT_GAME_HEIGHT' in locals() else 600
            game_x = (screen_width - game_width) // 2
            game_y = (screen_height - game_height) // 2
            self.game_region = (game_x, game_y, game_width, game_height)
    
    def capture_game_screen(self):
        try:
            screenshot = pyautogui.screenshot(region=self.game_region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"❌ Error capturing screen: {e}")
            return None
    
    def detect_fruits(self, frame):
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
    
    def slice_fruit(self, detection):
        try:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            abs_center_x = self.game_region[0] + center_x
            abs_center_y = self.game_region[1] + center_y
            
            slice_start_x = abs_center_x - SLICE_DISTANCE
            slice_start_y = abs_center_y - SLICE_DISTANCE
            slice_end_x = abs_center_x + SLICE_DISTANCE
            slice_end_y = abs_center_y + SLICE_DISTANCE
            
            pyautogui.moveTo(slice_start_x, slice_start_y, duration=MOUSE_MOVE_DURATION)
            pyautogui.dragTo(slice_end_x, slice_end_y, duration=MOUSE_DRAG_DURATION, button='left')
            
            print(f"🍎 Sliced fruit at ({center_x}, {center_y}) with confidence: {detection['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ Error slicing fruit: {e}")
    
    def auto_play_loop(self):
        global SHOW_DEBUG_WINDOW
        
        print("🎮 Auto-play system ready!")
        print("Press 't' + Enter to toggle auto-play on/off")
        print("Move mouse to corner to stop completely")
        
        frame_count = 0
        start_time = time.time()
        debug_error_count = 0
        last_fps_time = time.time()
        
        while self.is_running:
            try:
                if not self.toggle_manager.is_active:
                    time.sleep(0.1)
                    continue
                
                frame = self.capture_game_screen()
                if frame is None:
                    continue
                
                detections = self.detect_fruits(frame)
                
                current_time = time.time()
                for detection in detections:
                    if current_time - self.last_detection_time > self.min_detection_interval:
                        self.slice_fruit(detection)
                        self.last_detection_time = current_time
                
                if SHOW_DEBUG_WINDOW and frame_count % FRAME_DISPLAY_INTERVAL == 0:
                    try:
                        self.display_debug_frame(frame, detections)
                    except Exception as e:
                        debug_error_count += 1
                        if VERBOSE_LOGGING:
                            print(f"⚠️ Debug frame display error #{debug_error_count}: {e}")
                        
                        if debug_error_count >= 3:
                            SHOW_DEBUG_WINDOW = False
                            print("🔄 Debug window permanently disabled due to repeated errors")
                
                frame_count += 1
                time.sleep(SCREEN_CAPTURE_DELAY)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in auto-play loop: {e}")
                time.sleep(0.1)
        
        print("🛑 Auto-play stopped")
    
    def display_debug_frame(self, frame, detections):
        try:
            debug_frame = frame.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                
                if conf > 0.7:
                    color = (0, 255, 0)
                elif conf > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                
                text = f"{conf:.2f}"
                cv2.putText(debug_frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow("Fruit Detection Debug", debug_frame)
            cv2.waitKey(1)
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"⚠️ Error in debug frame display: {e}")
            print("🔄 Debug frame display failed, continuing without visual feedback")
    
    def cleanup(self):
        try:
            print("🧹 Cleaning up resources...")
            
            cv2.destroyAllWindows()
            time.sleep(0.1)
            
            for i in range(10):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def start_auto_play(self):
        self.is_running = True
        
        self.toggle_manager.start_monitoring()
        
        auto_play_thread = threading.Thread(target=self.auto_play_loop)
        auto_play_thread.daemon = True
        auto_play_thread.start()
        
        print("🎮 Auto-play system started!")
        print("Commands:")
        print("  't' + Enter - Toggle auto-play on/off")
        print("  'q' + Enter - Quit program")
        print("  Ctrl+C - Quit program")
        
        try:
            while self.is_running:
                if self.toggle_manager.quit_requested:
                    print("🔄 Quit requested, shutting down...")
                    break
                    
                time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupt received, shutting down...")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()
        
        self.is_running = False
        print("👋 Goodbye!")
        sys.exit(0)

def main():
    print("🍎 Fruit Ninja AI Auto-Player with Simple Toggle Control")
    print("=" * 55)
    
    try:
        ai_player = FruitNinjaAI()
        ai_player.start_auto_play()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()