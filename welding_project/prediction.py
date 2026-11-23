from ultralytics import YOLO

pred_model = YOLO("", task='detect') #best.pt

results = pred_model.predict(source = "datasets/test/images", conf = 0.50, save = True, stream= True)

for result in results:
    print(f"-------->{result.boxes}, {len(result.boxes)}")

    for box in result.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = pred_model.names()

        print(f"class : {class_name} || id: {class_id} || confidence: {confidence}\n")
