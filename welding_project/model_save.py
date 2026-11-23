from ultralytics import YOLO

best_model = YOLO("runs/detect/yolo_weld_training/weights/best.pt") # if traning got extended use that path

best_model.export( format = "engine", half = True, imgsz = 640) #nvidia CUDA compatible

best_model.export(format = "openvino", half = True)  #intel cpu and integrated gpu

best_model.export(format = "onnx", simplify = True)  #open neural network exchange

