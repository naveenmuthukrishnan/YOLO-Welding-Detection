from ultralytics import YOLO

val_model = YOLO("")  #best.pt after completion of traning under runs/detect/yolo_weld_traning

results = val_model.val(data = "data.yaml", imgsz = 640, split = "val")

print(results)
print(results.box.map50)
print(results.box.map)   #50-95% box overlapping