import streamlit as st
from PIL import Image
import numpy as np

# Delimiter to mark end of hidden message
DELIM = "1111111111111110"

# Convert message to binary
def message_to_bits(message: str) -> str:
    return ''.join(format(ord(c), '08b') for c in message)

# Convert binary to string
def bits_to_message(bits: str) -> str:
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    message = ""
    for c in chars:
        if len(c) == 8:
            message += chr(int(c, 2))
    return message

# Ensure image is RGB
def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

# LSB Encoding
def encode_lsb(image: Image.Image, message: str) -> Image.Image:
    img = to_rgb(image)
    data = np.array(img, dtype=np.int32)   # use int32 to avoid underflow/overflow
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
            flat[i, ch] = max(0, min(255, flat[i, ch]))  # clamp
            idx += 1
        if idx >= len(payload):
            break

    stego = flat.reshape(h, w, 3).astype(np.uint8)  # back to uint8
    return Image.fromarray(stego)

# LSB Decoding
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

# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="Steganography Forensics", layout="centered")
st.title("🔐 Steganography Forensics - Image LSB Tool")

mode = st.radio("Choose mode:", ["Encode", "Decode"])

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

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
