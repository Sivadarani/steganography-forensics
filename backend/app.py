from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, cv2, numpy as np, tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "steg_cnn.h5")
IMG_SIZE = 256

app = FastAPI(title="AI Steganalysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Model load failed:", e)
else:
    print("Model not found at", MODEL_PATH)

def preprocess_gray(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)/127.5 - 1.0
    return np.expand_dims(img, axis=(0,-1))

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse({"error": "Model not available. Train and place models/steg_cnn.h5"}, status_code=503)
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    x = preprocess_gray(img)
    p = float(model.predict(x, verbose=0)[0,0])
    return {"filename": file.filename, "stego_probability": p, "prediction": "STEGO" if p>=0.5 else "COVER"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
