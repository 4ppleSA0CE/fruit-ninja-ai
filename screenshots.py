import time
import mss
import os
from pynput import keyboard

# Define folder path
outputdir = os.path.join(os.path.expanduser('~'), "Documents", "fruit-ninja-ai", "screenshots")
os.makedirs(outputdir, exist_ok=True)

# Initialize mss for fast screenshots
sct = mss.mss()

taking_screenshots = False
i = 1

def on_press(key):
    global taking_screenshots
    try:
        if key.char == ';':
            taking_screenshots = not taking_screenshots
            if taking_screenshots:
                print("Taking screenshots...")
            else:
                print("Stopped taking screenshots.")
    except AttributeError:
        pass

# Start listening for key presses
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Main loop
while True:
    if taking_screenshots:
        # Capture screenshot using mss (much faster than pyautogui)
        screenshot = sct.grab(sct.monitors[1])  # Primary monitor
        if screenshot is not None:
            filename = os.path.join(outputdir, f"{i}.png")
            screenshot.save(filename)
            print(f"Screenshot {i} saved as {filename}")
            i += 1
        else:
            print("Failed to capture screenshot")