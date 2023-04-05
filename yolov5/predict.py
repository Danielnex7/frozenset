from ultralytics import YOLO

yolov5 = YOLO("models/yolov5s.pt")
# Запуск предсказания
yolov5.predict("ssdlite/short_test/000009.jpg", save=True, save_txt=True)