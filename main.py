from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import os

app = FastAPI()
model = YOLO("yolov8n.pt")  # pre-load sekali di startup

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # baca bytes langsung tanpa tulis ke disk
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # imgsz kecil, conf default cukup untuk ada/tidak-ada
    results = model(img, imgsz=640, conf=0.25)

    out = []
    names = model.names
    for r in results:
        for b in r.boxes:
            out.append({
                "label": names[int(b.cls[0])],
                "confidence": float(b.conf[0]),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()]
            })
    return {"detections": out}
