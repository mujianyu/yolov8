from ultralytics import YOLO
import cv2
# 加载模型
model = YOLO('yolov8n-seg.pt')
results = model('./images/multi-person.jpeg')
res = results[0].plot(boxes=False) #boxes=False表示不展示预测框，True表示同时展示预测框


cv2.imshow("YOLOv8 Inference", res,)
cv2.waitKey(0)