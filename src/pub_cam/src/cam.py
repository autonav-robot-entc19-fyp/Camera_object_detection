#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import math

# Initialize ROS node
rospy.init_node("object_detection_node")

# Initialize ROS publisher
pub = rospy.Publisher("/detected_objects", String, queue_size=10)

# Initialize CvBridge for converting between OpenCV and ROS Image
bridge = CvBridge()

# Start webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/jayabawan/Downloads/vehicles.mp4")
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Set up video writer
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while not rospy.is_shutdown():
    success, img = cap.read()
    results = model(img, stream=True)

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Class name
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            # Publish the detected class name to the ROS topic
            pub.publish(class_name)

    # Show the image
    cv2.imshow('sample', img)
    result.write(img)

    # Wait for a small amount of time to control the loop rate
    rospy.sleep(0.1)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
result.release()
cap.release()
cv2.destroyAllWindows()
