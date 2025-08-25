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
    # Default fallback values
    MODEL_PATH = "trained_models/best_fruit_detection_v11n.pt"
    CONFIDENCE_THRESHOLD = 0.3
    MOUSE_MOVE_DURATION = 0.05
    MOUSE_DRAG_DURATION = 0.1
    SLICE_DISTANCE = 20
    MIN_DETECTION_INTERVAL = 0.1
    FRAME_DISPLAY_INTERVAL = 30
    SCREEN_CAPTURE_DELAY = 0.01
    SHOW_DEBUG_WINDOW = True
    VERBOSE_LOGGING = True

# Safety: Move mouse to corner to stop
pyautogui.FAILSAFE = ENABLE_FAILSAFE if 'ENABLE_FAILSAFE' in locals() else True
pyautogui.PAUSE = MOUSE_PAUSE if 'MOUSE_PAUSE' in locals() else 0.01

class HotkeyManager:
    """Manages global hotkeys for the application."""
    
    def __init__(self):
        self.is_active = False
        self.command_pressed = False
        self.four_pressed = False
        self.s_pressed = False
        self.hotkey_thread = None
        
    def start_monitoring(self):
        """Start monitoring for hotkeys in a separate thread."""
        self.hotkey_thread = threading.Thread(target=self._monitor_hotkeys, daemon=True)
        self.hotkey_thread.start()
        print("⌨️ Hotkey monitoring started: Press Cmd+4+S to toggle auto-play")
    
    def _monitor_hotkeys(self):
        """Monitor for hotkey combinations."""
        try:
            from pynput import keyboard
            
            def on_press(key):
                try:
                    # Check for Command key (meta key on macOS)
                    if key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                        self.command_pressed = True
                    # Check for 4 key
                    elif hasattr(key, 'char') and key.char == '4':
                        self.four_pressed = True
                    # Check for S key
                    elif hasattr(key, 'char') and key.char.lower() == 's':
                        self.s_pressed = True
                    
                    # Check if hotkey combination is complete
                    if self.command_pressed and self.four_pressed and self.s_pressed:
                        self.is_active = not self.is_active
                        status = "ACTIVATED" if self.is_active else "DEACTIVATED"
                        print(f"\n🎯 Auto-play {status} by hotkey!")
                        
                        # Reset key states
                        self.command_pressed = False
                        self.four_pressed = False
                        self.s_pressed = False
                        
                except AttributeError:
                    pass
            
            def on_release(key):
                try:
                    # Reset key states when keys are released
                    if key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                        self.command_pressed = False
                    elif hasattr(key, 'char') and key.char == '4':
                        self.four_pressed = False
                    elif hasattr(key, 'char') and key.char.lower() == 's':
                        self.s_pressed = False
                except AttributeError:
                    pass
            
            # Start listening for key events
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
                
        except ImportError:
            print("⚠️ pynput not available, using fallback hotkey system")
            print("Install with: pip3 install pynput")
            self._fallback_hotkey_monitoring()
    
    def _fallback_hotkey_monitoring(self):
        """Fallback hotkey monitoring using input() for when pynput is not available."""
        def fallback_thread():
            while True:
                try:
                    key = input("Press 't' + Enter to toggle auto-play, 'q' + Enter to quit: ").strip().lower()
                    if key == 't':
                        self.is_active = not self.is_active
                        status = "ACTIVATED" if self.is_active else "DEACTIVATED"
                        print(f"🎯 Auto-play {status}!")
                    elif key == 'q':
                        break
                except (EOFError, KeyboardInterrupt):
                    break
        
        fallback_thread = threading.Thread(target=fallback_thread, daemon=True)
        fallback_thread.start()

class FruitNinjaAI:
    def __init__(self):
        self.model = None
        self.is_running = False
        self.game_region = None
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.last_detection_time = 0
        self.min_detection_interval = MIN_DETECTION_INTERVAL
        
        # Initialize hotkey manager
        self.hotkey_manager = HotkeyManager()
        
        # Load the trained model
        self.load_model()
        
        # Initialize mouse control
        self.setup_mouse_control()
    
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
    
    def setup_mouse_control(self):
        """Setup mouse control parameters."""
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        print(f"🖥️ Screen resolution: {screen_width}x{screen_height}")
        
        # Default game region (center of screen, adjust as needed)
        game_width = DEFAULT_GAME_WIDTH if 'DEFAULT_GAME_WIDTH' in locals() else 800
        game_height = DEFAULT_GAME_HEIGHT if 'DEFAULT_GAME_HEIGHT' in locals() else 600
        game_x = (screen_width - game_width) // 2
        game_y = (screen_height - game_height) // 2
        
        self.game_region = (game_x, game_y, game_width, game_height)
        print(f"🎮 Default game region: {self.game_region}")
        
        # Ask user to adjust region if needed
        self.adjust_game_region()
    
    def adjust_game_region(self):
        """Allow user to adjust the game region."""
        print("\n🎯 Game Region Setup:")
        print("1. Open Fruit Ninja in a window")
        print("2. Position it where you want to play")
        print("3. Press Enter when ready")
        
        input("Press Enter to continue...")
        
        print("Click and drag to select the game area...")
        print("Move mouse to top-left corner of game window, then press Enter")
        input("Position mouse at top-left corner and press Enter...")
        
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
        
        self.game_region = (x, y, width, height)
        print(f"✅ Game region set to: {self.game_region}")
        
        # Confirm region
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
        """Capture the game screen region."""
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
    
    def slice_fruit(self, detection):
        """Perform a slicing motion on the detected fruit."""
        try:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # Convert relative coordinates to absolute screen coordinates
            abs_center_x = self.game_region[0] + center_x
            abs_center_y = self.game_region[1] + center_y
            
            # Calculate slice endpoints (diagonal slice)
            slice_start_x = abs_center_x - SLICE_DISTANCE
            slice_start_y = abs_center_y - SLICE_DISTANCE
            slice_end_x = abs_center_x + SLICE_DISTANCE
            slice_end_y = abs_center_y + SLICE_DISTANCE
            
            # Perform the slice motion
            pyautogui.moveTo(slice_start_x, slice_start_y, duration=MOUSE_MOVE_DURATION)
            pyautogui.dragTo(slice_end_x, slice_end_y, duration=MOUSE_DRAG_DURATION, button='left')
            
            print(f"🍎 Sliced fruit at ({center_x}, {center_y}) with confidence: {detection['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ Error slicing fruit: {e}")
    
    def auto_play_loop(self):
        """Main auto-play loop."""
        print("🎮 Auto-play system ready!")
        print("Press Cmd+4+S to toggle auto-play on/off")
        print("Move mouse to corner to stop completely")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            try:
                # Check if hotkey has activated auto-play
                if not self.hotkey_manager.is_active:
                    time.sleep(0.1)
                    continue
                
                # Capture game screen
                frame = self.capture_game_screen()
                if frame is None:
                    continue
                
                # Detect fruits
                detections = self.detect_fruits(frame)
                
                # Process detections
                current_time = time.time()
                for detection in detections:
                    # Check if enough time has passed since last detection
                    if current_time - self.last_detection_time > self.min_detection_interval:
                        self.slice_fruit(detection)
                        self.last_detection_time = current_time
                
                # Display frame with detections (optional, for debugging)
                if SHOW_DEBUG_WINDOW and frame_count % FRAME_DISPLAY_INTERVAL == 0:
                    self.display_debug_frame(frame, detections)
                
                frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(SCREEN_CAPTURE_DELAY)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in auto-play loop: {e}")
                time.sleep(0.1)
        
        print("🛑 Auto-play stopped")
    
    def display_debug_frame(self, frame, detections):
        """Display frame with detection overlays for debugging."""
        debug_frame = frame.copy()
        
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
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence text
            text = f"{conf:.2f}"
            cv2.putText(debug_frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show frame
        cv2.imshow("Fruit Detection Debug", debug_frame)
        cv2.waitKey(1)
    
    def start_auto_play(self):
        """Start the auto-play system."""
        self.is_running = True
        
        # Start hotkey monitoring
        self.hotkey_manager.start_monitoring()
        
        # Start auto-play in a separate thread
        auto_play_thread = threading.Thread(target=self.auto_play_loop)
        auto_play_thread.daemon = True
        auto_play_thread.start()
        
        print("🎮 Auto-play system started!")
        print("Commands:")
        print("  Cmd+4+S - Toggle auto-play on/off")
        print("  Move mouse to corner - Stop completely")
        print("  Ctrl+C - Quit program")
        
        # Main loop for user input
        try:
            while True:
                time.sleep(0.1)  # Keep main thread alive
                    
        except KeyboardInterrupt:
            pass
        
        # Cleanup
        self.is_running = False
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

def main():
    """Main function."""
    print("🍎 Fruit Ninja AI Auto-Player with Hotkey Control")
    print("=" * 50)
    
    try:
        # Create and start the AI player
        ai_player = FruitNinjaAI()
        ai_player.start_auto_play()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()