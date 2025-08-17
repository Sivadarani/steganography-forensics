# Steganography Forensics — Web App (Streamlit)

This app demonstrates **encode**, **decode**, and **forensic detection** for image steganography using **LSB**.

## How to run (Windows)

1) Install Python 3.10+
2) Open Command Prompt and run:
```
pip install -r requirements.txt
streamlit run app.py
```
3) A browser will open at `http://localhost:8501`

## Tabs
- **Encode**: hide a secret message inside an image (PNG recommended)
- **Decode**: extract a message from an image created by this encoder
- **Forensics**: visualize the LSB plane and show basic stats (randomness heuristic)

## Notes
- For JPEGs, the app will still work, but **PNG** is more reliable (lossless).
- This is a baseline. You can extend the Forensics tab with CNN/AI for better detection of unknown algorithms.
