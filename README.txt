# Stego Web App Starter

## Structure
```
stego_webapp_starter/
 ├── frontend/
 │   └── index.html          # Simple UI (open in browser)
 └── backend/
     ├── app.py              # FastAPI server
     ├── requirements.txt
     └── models/
         └── steg_cnn.h5     # Put your trained model here
```

## How to run (backend)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
This starts the API at http://localhost:8000

## How to use (frontend)
- Open `frontend/index.html` in your browser.
- Upload an image and click **Analyze**.
- The page will call POST /analyze on the backend and display prediction.

## Train a model
Use your own pipeline (or the one provided earlier) to produce `models/steg_cnn.h5`.
Place that file under `backend/models/`.
