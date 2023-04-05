from ultralytics import YOLO

yolov5 = YOLO("yolov5/best.pt")
# Запуск предсказания
yolov5.predict("Examples/am3_1_frame004.jpg", save=True, save_txt=True)