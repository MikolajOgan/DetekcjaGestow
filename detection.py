import cv2
import torch
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Path to your model
model_path = r"C:\Users\mikol\Documents\p40\best.pt"

# Verify model path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=True)

# Initialize webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot access the webcam")
    exit()

print("Press 'q' to exit the webcam feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Run inference
    results = model(frame)

    # Render results
    output_frame = results.render()[0]

    # Display the output
    cv2.imshow('YOLOv5 Real-Time Object Detection', output_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
