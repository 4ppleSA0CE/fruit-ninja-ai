import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("path/to/your/trained_model.pt")

# Define the region of the game screen to capture (adjust as needed)
region = (100, 100, 800, 600)  # (x, y, width, height)

while True:
    # Capture the game screen
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)  # Convert to NumPy array
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Run YOLO inference
    results = model(frame)

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class ID

            if conf > 0.5:  # Check if confidence is greater than 50%
                # Calculate the center of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Move the mouse to the center of the bounding box
                pyautogui.moveTo(region[0] + center_x, region[1] + center_y, duration=0.1)

                # Simulate a slicing motion
                pyautogui.dragTo(region[0] + x2, region[1] + y2, duration=0.2, button='left')

    # Display the frame
    cv2.imshow("Game Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()