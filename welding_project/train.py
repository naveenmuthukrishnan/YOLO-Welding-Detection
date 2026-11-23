from ultralytics import YOLO

model = YOLO("yolo11s.pt")  #s--> small model (n,s,m,l)

model.train(
    data = "data.yaml",
    batch = 6,
    epochs = 200,
    device = 0,  #triggers CUDA GPU
    patience = 50,
    imgsz = 640,   #increase the image size that matches ur camera feed size
    degrees = 20,
    #flipud = 0.50,
    fliplr = 0.20,
    scale = 0.50,
    perspective = 0.0005,
    hsv_h = 0.015,
    hsv_s = 0.5,
    hsv_v = 0.5,
    mosaic = 1.0,
    close_mosaic = 10,
    mixup = 0.1,

    project = "runs/detect/",
    name = "yolo_weld_training"
) # based on your dataset fix the parameters



#Continue with last.pt based on metrics if traning needed