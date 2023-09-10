import cv2
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt', task='segment')
video_path = "./videos/mother_wx.mp4"
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()