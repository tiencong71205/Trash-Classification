from ultralytics import YOLO
model = YOLO("3.refine_last_phase3_10epochs.pt")  # model .pt của YOLOv8
model.export(format="onnx", opset=12, simplify=True)
