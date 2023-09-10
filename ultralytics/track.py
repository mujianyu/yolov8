from ultralytics import YOLO

model = YOLO('yolov8n.pt',task='detect')
results = model.track(source="./videos/mother_wx.mp4", show=True)