#!/usr/bin/env python3
"""
Configuration file for Fruit Ninja AI Auto-Player
Adjust these settings to customize the behavior
"""

# Model settings
MODEL_PATH = "trained_models/best_fruit_detection_v11n.pt"
CONFIDENCE_THRESHOLD = 0.2  # Lowered for more detections

# Game region settings
DEFAULT_GAME_WIDTH = 800
DEFAULT_GAME_HEIGHT = 600

# Mouse control settings
MOUSE_MOVE_DURATION = 0.01  # Very fast mouse movement
MOUSE_DRAG_DURATION = 0.02  # Very fast slice motion
SLICE_DISTANCE = 30          # Larger slice distance for better effect

# Performance settings
MIN_DETECTION_INTERVAL = 0.02  # Much faster detection response
FRAME_DISPLAY_INTERVAL = 60    # Show debug frame less often
SCREEN_CAPTURE_DELAY = 0.001   # Minimal delay for maximum speed

# Safety settings
ENABLE_FAILSAFE = True        # Move mouse to corner to stop
MOUSE_PAUSE = 0.001          # Minimal mouse delay

# Debug settings
SHOW_DEBUG_WINDOW = False     # Set to False to completely avoid OpenCV display errors
VERBOSE_LOGGING = True        # Print detailed logs
