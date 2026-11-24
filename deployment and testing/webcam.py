import time
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best_weld.pt", task= "detect")  #best.pt from traning
model(np.zeros((1024,1024,3), dtype= np.uint8), verbose = False) #during warmup it runs some black frames

capture = cv2.VideoCapture("weld_2.mp4") #either pass cam or .mp4 file as str("weld.mp4")

good_id = {1}
defective_id = {0,2}

while capture.isOpened():
    start = time.perf_counter()
    success, frame = capture.read()

    if not success:
        break
    frame_has_defect = False
    frame_has_good = False

    frame = cv2.resize(frame, dsize=(1024,1024))
    results = model(frame, conf = 0.25, imgsz = 1024, device = 0) #change device according!!!

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_id in defective_id:
                frame_has_defect = True
                color = (0,0,255) #red color
            elif class_id in good_id:
                frame_has_good = True
                color = (0,255,0) #green color
            else:
                color = (200,200,200)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1-10,20)), cv2.FONT_HERSHEY_SIMPLEX,0.7, color, 3)


    #verdict
    if frame_has_defect:
        verdict, vcolor = "Defective", (0,0,255)
    elif frame_has_good:
        verdict, vcolor = "Good", (0,255,0)
    else:
        verdict, vcolor = "NO Weld", (200,200,200)

    cv2.putText(frame,f"verdict: {verdict}", (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7, vcolor,3)

    #FPS
    fps = 1.0 / (time.perf_counter() - start)
    cv2.putText(frame, f"fps: {fps:.1f}", (500, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),3)

    cv2.imshow("Detection Window", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113: #q or Q to exit
        break

capture.release()
cv2.destroyAllWindows()