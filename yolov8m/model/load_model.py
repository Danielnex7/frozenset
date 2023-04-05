from ultralytics import YOLO
model = YOLO("yolov8m_71ep_helm.pt") 
results = model.predict("frozenset/ssdlite/short_test", save=True, save_txt=True)
