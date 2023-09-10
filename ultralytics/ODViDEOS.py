import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
# 打开文件
video_path = "./videos/cxk.mp4"
cap = cv2.VideoCapture(video_path)
# 循环播放视频
while cap.isOpened():
    # 读取视频的每一帧
    success, frame = cap.read()

    if success:
        # 运行yolov8
        results = model(frame)
        # 可视化结果
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # 按住q键盘播放结束
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #结束播放则循环结束
        break

# 释放捕捉和窗口
cap.release()
cv2.destroyAllWindows()