from ultralytics import YOLO

best_model = YOLO("runs/detect/yolo_weld_training/weights/best.pt") # if traning got extended use that path

best_model.export( format = "engine", half = True, imgsz = 1024, dynamic = True) #nvidia CUDA compatible

best_model.export(format = "openvino", half = True, dynamic = True, imgsz = 1024)  #intel cpu and integrated gpu

best_model.export(format = "onnx", dynamic = True, imgsz = 1024)  #open neural network exchange

