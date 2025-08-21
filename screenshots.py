import time
import pyautogui
import os
from pynput import keyboard

# Define folder path
outputdir = os.path.join(os.path.expanduser('~'), "Documents", "fruitnin", "screenshots")
os.makedirs(outputdir, exist_ok=True)

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
        screenshot = pyautogui.screenshot()
        filename = os.path.join(outputdir, f"{i}.png")
        screenshot.save(filename)
        print(f"Screenshot {i} saved as {filename}")
        i += 1
        time.sleep(0.1)