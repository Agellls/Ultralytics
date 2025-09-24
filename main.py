from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil, os

app = FastAPI()
model = YOLO("yolov8n.pt")  # model ringan (nano) bawaan COCO

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Simpan file sementara
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Jalankan deteksi
    results = model(temp_path, imgsz=640, conf=0.25)
    os.remove(temp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            cls = model.names[int(box.cls[0])]
            detections.append({
                "label": cls,
                "confidence": float(box.conf[0]),
                "xyxy": box.xyxy[0].tolist()
            })
    return {"detections": detections}
