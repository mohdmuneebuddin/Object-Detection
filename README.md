# Object-Detection
object detection with live default camera by using yolov8 
import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize webcam (default camera)
cap = cv2.VideoCapture(0)

    while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

     results = model(frame)

     # Plot bounding boxes on the frame
     bounding_boxes = results[0].plot()

     # Display the frame
     cv2.imshow("YOLOv8 Default camera Object Detection", bounding_boxes)

     if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
