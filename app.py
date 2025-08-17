import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ====== Constants ======
DELIM = [0] * 16 + [1] * 16  # delimiter to mark end of message


# ====== Helper Functions ======
def to_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is RGB"""
    return img.convert("RGB")


def message_to_bits(msg: str) -> list:
    bits = []
    for c in msg:
        binval = bin(ord(c))[2:].rjust(8, '0')
        bits.extend([int(b) for b in binval])
    return bits


def bits_to_message(bits: list) -> str:
    chars = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b + 8]
        if len(byte) < 8:
            continue
        chars.append(chr(int("".join([str(x) for x in byte]), 2)))
    return "".join(chars)


# ====== Steganography Functions ======
def encode_lsb(image: Image.Image, message: str) -> Image.Image:
    img = to_rgb(image)
    # ✅ Ensure always uint8 to avoid -2 out of bounds
    data = np.array(img, dtype=np.uint8).copy()
    h, w, _ = data.shape
    capacity_bits = h * w * 3
    payload = message_to_bits(message) + DELIM
    if len(payload) > capacity_bits:
        raise ValueError(f"Message too long for this image. Capacity ~{capacity_bits // 8} bytes, need {len(payload) // 8} bytes.")
    idx = 0
    flat = data.reshape(-1, 3)
    for i in range(flat.shape[0]):
        for ch in range(3):
            if idx >= len(payload):
                break
            flat[i, ch] = (flat[i, ch] & ~1) | int(payload[idx])  # safe uint8
            idx += 1
        if idx >= len(payload):
            break
    stego = flat.reshape(h, w, 3)
    return Image.fromarray(stego.astype(np.uint8))


def decode_lsb(image: Image.Image) -> str:
    img = to_rgb(image)
    data = np.array(img, dtype=np.uint8)
    flat = data.reshape(-1, 3)
    bits = []
    for i in range(flat.shape[0]):
        for ch in range(3):
            bits.append(flat[i, ch] & 1)
    # find delimiter
    for i in range(len(bits)):
        if bits[i:i + len(DELIM)] == DELIM:
            msg_bits = bits[:i]
            return bits_to_message(msg_bits)
    return ""


# ====== Streamlit App ======
st.title("🔐 Steganography Forensics - Image LSB Tool")

menu = ["Encode", "Decode", "About"]
choice = st.sidebar.selectbox("Select Mode", menu)

if choice == "Encode":
    st.subheader("Embed a secret message into an image")
    cover_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"])
    secret_msg = st.text_area("Enter the secret message")
    if cover_file and secret_msg:
        cover = Image.open(cover_file)
        st.image(cover, caption="Original Cover Image", use_container_width=True)
        if st.button("Encode"):
            try:
                stego = encode_lsb(cover, secret_msg)
                st.image(stego, caption="Stego Image", use_container_width=True)
                stego.save("stego.png")
                st.success("Message encoded successfully! Stego image saved as stego.png")
                with open("stego.png", "rb") as f:
                    st.download_button("Download Stego Image", f, file_name="stego.png", mime="image/png")
            except Exception as e:
                st.error(f"Encoding failed: {e}")

elif choice == "Decode":
    st.subheader("Extract a hidden message from an image")
    stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])
    if stego_file:
        stego = Image.open(stego_file)
        st.image(stego, caption="Stego Image", use_container_width=True)
        if st.button("Decode"):
            try:
                message = decode_lsb(stego)
                st.success("Decoded message:")
                st.code(message)
            except Exception as e:
                st.error(f"Decoding failed: {e}")

else:
    st.subheader("About")
    st.markdown("""
    This is a **Steganography Forensics** app built with Streamlit.  
    - Supports **LSB Encoding/Decoding** for images.  
    - Upload an image, hide a message, then extract it back.  
    - Future versions can support **audio, video, and medical images**.  
    """)

