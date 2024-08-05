import numpy as np
import cv2
import torch
import os
import datetime
import sys
import models
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

# Redirect stderr to stdout
sys.stderr = sys.stdout

# Load YOLOv5 model
model_path = 'C:/yolov5x/yolov5/runs/train/exp/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True).to(device)

# Set initial parameters for visualization
MARGIN = 10
ROW_SIZE = 20
FONT_SIZE = 0.8  # Adjust font size here (larger size)
FONT_THICKNESS = 2  # Adjust font thickness here (thicker)
TEXT_COLOR = (0, 255, 0)  # Adjust text color here (green)
YELLOW_COLOR = (0, 255, 255)  # Yellow color for potential errors or close objects

# Directory to save detected images
SAVE_DIR = "C:/yolov5x/yolov5/result"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Poppins font
font_path = "C:/yolov5x/yolov5/Fonts/Poppins-Regular.ttf"  # Replace with your actual font file path
font = cv2.FONT_HERSHEY_SIMPLEX

# Function to visualize detections and save the detected image
def visualize_and_save(image, results, total_objects=100, conf_threshold=0.5, save_dir=None):
    if isinstance(results, models.common.Detections):
        detections = results.xyxy[0].cpu().numpy()
        object_count = 0

        # Lists to store detected boxes for each object type
        detected_boxes = []
        object_types = []

        for i, detection in enumerate(detections, 1):
            xmin, ymin, xmax, ymax = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
            confidence = float(detection[4])
            
            if confidence >= conf_threshold:
                object_count += 1
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue box for detected objects
                text_location = (xmin + MARGIN, ymin + ROW_SIZE)
                cv2.putText(image, str(object_count), text_location, font, FONT_SIZE * 0.5, (0, 0, 255), FONT_THICKNESS, lineType=cv2.LINE_AA)
                detected_boxes.append((xmin, ymin, xmax, ymax))
                object_types.append(i)  # Store the object type (index)

        detection_text = "Detected Image (Captured)"
        cv2.putText(image, detection_text, (10, 60), font, FONT_SIZE * 1.2, TEXT_COLOR, FONT_THICKNESS // 1, lineType=cv2.LINE_AA)
        cv2.putText(image, f'Objects detected: {object_count}', (10, 30), font, FONT_SIZE * 1.2, TEXT_COLOR, FONT_THICKNESS // 1, lineType=cv2.LINE_AA)

        # Save the image if a save directory is provided
        if save_dir is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            image_name = f"detected_image_{timestamp}.jpg"
            image_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_path, image)
            print(f"Detected image saved as {image_path}")
    else:
        raise TypeError(f"Unsupported results type: {type(results)}")



def analyze_close_objects(image, detected_boxes, threshold):
    """
    Analyze and mark objects that are close to each other.
    :param image: Image to draw on
    :param detected_boxes: List of detected boxes (xmin, ymin, xmax, ymax)
    :param threshold: Distance threshold for marking close objects
    """
    for i, box1 in enumerate(detected_boxes):
        xmin1, ymin1, xmax1, ymax1 = box1
        for j, box2 in enumerate(detected_boxes):
            if i >= j:
                continue
            xmin2, ymin2, xmax2, ymax2 = box2
            center1 = ((xmin1 + xmax1) // 2, (ymin1 + ymax1) // 2)
            center2 = ((xmin2 + xmax2) // 2, (ymin2 + ymax2) // 2)
            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
            if distance < threshold:
                cv2.rectangle(image, (xmin1, ymin1), (xmax1, ymax1), YELLOW_COLOR, 2)  # Yellow box for potential errors
                cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), YELLOW_COLOR, 2)  # Yellow box for potential errors
                cv2.putText(image, "Close", (xmin1, ymin1 - 10), font, FONT_SIZE * 0.5, YELLOW_COLOR, FONT_THICKNESS, lineType=cv2.LINE_AA)

# Capture video from webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FPS, 30)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 820)

# Create OpenCV windows
cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera View", 1080, 820)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Objects", 1080, 820)

# Create trackbars for alpha and beta adjustments
cv2.createTrackbar('Alpha', 'Camera View', 10, 30, lambda x: None)
cv2.createTrackbar('Beta', 'Camera View', 100, 200, lambda x: None)

# Variable to store captured image
captured_image = None

# Main loop
while True:
    success, frame = webcam.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    alpha = cv2.getTrackbarPos('Alpha', 'Camera View') / 10.0
    beta = cv2.getTrackbarPos('Beta', 'Camera View') - 100
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Display frame in "Camera View"
    cv2.imshow("Camera View", frame)

    # Capture and process image when 's' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        captured_image = frame.copy()
        print("Image captured.")
        if captured_image is not None:
            results_captured = model(captured_image)
            detected_frame = captured_image.copy()
            visualize_and_save(detected_frame, results_captured, save_dir=SAVE_DIR)
            cv2.imshow("Detected Objects", detected_frame)

    # Display the last detected image in "Detected Objects"
    if captured_image is None:
        cv2.imshow("Detected Objects", np.zeros_like(frame))  # Show blank screen if no image captured

    # Exit loop if 'q' key is pressed or any window is closed
    if cv2.getWindowProperty("Camera View", cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty("Detected Objects", cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting...")
        break

# Release webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
