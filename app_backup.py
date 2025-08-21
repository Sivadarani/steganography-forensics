import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------- LSB Steganography ----------------------

DELIM = "1111111111111110"

def message_to_bits(message: str) -> str:
    return ''.join(format(ord(c), '08b') for c in message)

def bits_to_message(bits: str) -> str:
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    message = ""
    for c in chars:
        if len(c) == 8:
            message += chr(int(c, 2))
    return message

def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

def encode_lsb(image: Image.Image, message: str) -> Image.Image:
    img = to_rgb(image)
    data = np.array(img, dtype=np.int32)
    h, w, _ = data.shape
    capacity_bits = h * w * 3
    payload = message_to_bits(message) + DELIM
    if len(payload) > capacity_bits:
        raise ValueError(
            f"Message too long for this image. Capacity ~{capacity_bits//8} bytes, need {len(payload)//8} bytes."
        )
    idx = 0
    flat = data.reshape(-1, 3)
    for i in range(flat.shape[0]):
        for ch in range(3):
            if idx >= len(payload):
                break
            bit = int(payload[idx])
            flat[i, ch] = (flat[i, ch] & ~1) | bit
            flat[i, ch] = max(0, min(255, flat[i, ch]))
            idx += 1
        if idx >= len(payload):
            break
    stego = flat.reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(stego)

def decode_lsb(image: Image.Image) -> str:
    img = to_rgb(image)
    data = np.array(img, dtype=np.uint8)
    bits = ""
    flat = data.reshape(-1, 3)
    for i in range(flat.shape[0]):
        for ch in range(3):
            bits += str(flat[i, ch] & 1)
            if bits.endswith(DELIM):
                return bits_to_message(bits[:-len(DELIM)])
    return bits_to_message(bits)

# ---------------------- Stego Detection Model ----------------------

stego_model = models.resnet18(weights=None)
stego_model.fc = nn.Linear(stego_model.fc.in_features, 2)  # 2 classes: Clean / Stego
stego_model.load_state_dict(torch.load("models/best_stego_resnet18.pth", map_location='cpu'))
stego_model.eval()

def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def predict_stego(image: Image.Image) -> str:
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = stego_model(image)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
        label = "Clean" if pred.item() == 0 else "Stego"
        confidence = probs[0][pred.item()].item() * 100
        return label, confidence

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Steganography Forensics", layout="centered")
st.title("🔐 Steganography Forensics - Image LSB Tool")

mode = st.radio("Choose mode:", ["Encode", "Decode", "Chat Stego Detector"])

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Chatbot history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if mode == "Encode":
        message = st.text_area("Enter the secret message:")
        if st.button("Encode"):
            if message.strip() == "":
                st.error("Message cannot be empty.")
            else:
                try:
                    stego_img = encode_lsb(image, message)
                    st.image(stego_img, caption="Stego Image", use_container_width=True)
                    stego_img.save("stego.png")
                    with open("stego.png", "rb") as f:
                        st.download_button("Download Stego Image", f, file_name="stego.png")
                except Exception as e:
                    st.error(f"Encoding failed: {e}")

    elif mode == "Decode":
        if st.button("Decode"):
            try:
                secret = decode_lsb(image)
                if secret.strip() == "":
                    st.warning("No hidden message found.")
                else:
                    st.success("Hidden message extracted:")
                    st.code(secret)
            except Exception as e:
                st.error(f"Decoding failed: {e}")

    elif mode == "Chat Stego Detector":
        user_input = st.text_input("Ask: Is there hidden data in this image?")
        if st.button("Send"):
            if user_input.strip() != "":
                label, confidence = predict_stego(image)
                bot_reply = f"The image is likely **{label}** with confidence {confidence:.2f}%."
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", bot_reply))

        # Display chat
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")
