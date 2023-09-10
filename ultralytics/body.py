from ultralytics import YOLO
import cv2
model = YOLO('yolov8n-pose.pt')
results = model('./images/multi-person.jpeg')
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)