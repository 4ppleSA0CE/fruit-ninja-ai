#!/usr/bin/env python3
"""
Configuration file for Fruit Ninja AI Auto-Player
Adjust these settings to customize the behavior
"""

# Model settings
MODEL_PATH = "trained_models/best_fruit_detection_v11n.pt"
CONFIDENCE_THRESHOLD = 0.3  # Lowered from 0.6 to catch more detections

# Game region settings
DEFAULT_GAME_WIDTH = 800
DEFAULT_GAME_HEIGHT = 600

# Mouse control settings
MOUSE_MOVE_DURATION = 0.05  # Duration for mouse movement (seconds)
MOUSE_DRAG_DURATION = 0.1   # Duration for slicing motion (seconds)
SLICE_DISTANCE = 20          # Distance for slice motion (pixels)

# Performance settings
MIN_DETECTION_INTERVAL = 0.1  # Minimum time between detections (seconds)
FRAME_DISPLAY_INTERVAL = 30   # Show debug frame every N frames
SCREEN_CAPTURE_DELAY = 0.01   # Delay between screen captures (seconds)

# Safety settings
ENABLE_FAILSAFE = True        # Move mouse to corner to stop
MOUSE_PAUSE = 0.01           # Delay between mouse actions

# Debug settings
SHOW_DEBUG_WINDOW = True      # Show detection overlay window
VERBOSE_LOGGING = True        # Print detailed logs
