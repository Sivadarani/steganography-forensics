import io
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# ---- Utility functions ----
DELIM = '1111111111111110'  # end-of-message marker

def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def message_to_bits(msg: str) -> str:
    return ''.join(format(ord(c), '08b') for c in msg)

def bits_to_message(bits: str) -> str:
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    out = []
    for c in chars:
        if len(c) < 8:
            continue
        try:
            out.append(chr(int(c, 2)))
        except ValueError:
            pass
    return ''.join(out)

def encode_lsb(image: Image.Image, message: str) -> Image.Image:
    img = to_rgb(image)
    data = np.array(img, dtype=np.uint8).copy()   # ensure uint8 safe copy
    h, w, _ = data.shape
    capacity_bits = h * w * 3
    payload = message_to_bits(message) + DELIM
    if len(payload) > capacity_bits:
        raise ValueError(f"Message too long for this image. Capacity ~{capacity_bits//8} bytes, need {len(payload)//8} bytes.")
    
    idx = 0
    flat = data.reshape(-1, 3)
    for i in range(flat.shape[0]):
        for ch in range(3):
            if idx >= len(payload):
                break
            flat[i, ch] = (flat[i, ch] & ~1) | int(payload[idx])
            idx += 1
        if idx >= len(payload):
            break

    stego = flat.reshape(h, w, 3)
    stego = np.clip(stego, 0, 255).astype(np.uint8)   # FIX: prevent -2 or 256 values
    return Image.fromarray(stego)

def decode_lsb(image: Image.Image) -> str:
    img = to_rgb(image)
    data = np.array(img, dtype=np.uint8)
    bits = ''.join(str(px & 1) for px in data.flatten())
    # split at delimiter
    cut = bits.find(DELIM)
    if cut == -1:
        return ""
    msg_bits = bits[:cut]
    return bits_to_message(msg_bits)

def lsb_plane(image: Image.Image) -> np.ndarray:
    img = to_rgb(image)
    data = np.array(img, dtype=np.uint8)
    return (data & 1)

def lsb_stats(lsb: np.ndarray) -> dict:
    total = lsb.size
    ones = int(np.sum(lsb))
    zeros = int(total - ones)
    ratio = ones / total if total else 0.0
    verdict = "Suspicious (≈ random LSBs)" if 0.48 <= ratio <= 0.52 else "Likely clean / inconclusive"
    return {
        "total_bits": total,
        "ones": ones,
        "zeros": zeros,
        "proportion_ones": ratio,
        "verdict": verdict
    }

# ---- Streamlit UI ----
st.set_page_config(page_title="Steganography Forensics", page_icon="🕵️", layout="wide")
st.title("🕵️ Steganography Forensics — Encode • Decode • Detect")

tabs = st.tabs(["🔐 Encode (Hide Message)", "📬 Decode (Extract Message)", "🧪 Forensics (Detect)"])

with tabs[0]:
    st.subheader("Hide a secret message inside an image (LSB)")
    up = st.file_uploader("Upload a cover image (PNG/JPG recommended PNG for lossless)", type=["png", "jpg", "jpeg"])
    msg = st.text_area("Secret message", placeholder="Type the message you want to hide...")
    colA, colB = st.columns(2)
    with colA:
        if up:
            cover = Image.open(up)
            st.image(cover, caption=f"Cover image — {cover.size[0]}x{cover.size[1]}", use_container_width=True)  # updated
    with colB:
        if up and msg:
            if st.button("Encode message"):
                try:
                    stego = encode_lsb(cover, msg)
                    buf = io.BytesIO()
                    stego.save(buf, format="PNG")
                    st.image(stego, caption="Stego image (preview)", use_container_width=True)  # updated
                    st.download_button("⬇️ Download stego image", data=buf.getvalue(), file_name="stego_output.png", mime="image/png")
                    st.success("Message embedded successfully into image.")
                except Exception as e:
                    st.error(f"Encoding failed: {e}")

with tabs[1]:
    st.subheader("Extract hidden message from a stego image")
    up2 = st.file_uploader("Upload a suspected stego image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="dec_upl")
    if up2:
        stego_img = Image.open(up2)
        st.image(stego_img, caption="Input image", use_container_width=True)  # updated
        if st.button("Decode message"):
            text = decode_lsb(stego_img)
            if text:
                st.success("Message found!")
                st.code(text, language="text")
            else:
                st.warning("No message detected with this simple LSB decoder. (It may not use LSB or could be encrypted.)")

with tabs[2]:
    st.subheader("Forensic detection — LSB plane analysis")
    up3 = st.file_uploader("Upload an image to analyze", type=["png", "jpg", "jpeg"], key="for_upl")
    if up3:
        img = Image.open(up3)
        st.image(img, caption="Image under test", use_container_width=True)  # updated

        lsb = lsb_plane(img)
        stats = lsb_stats(lsb)

        # Show stats
        st.write("**LSB Statistics**")
        st.json({
            "Total bits analyzed": stats["total_bits"],
            "LSB 1s": stats["ones"],
            "LSB 0s": stats["zeros"],
            "Proportion of LSB 1s": round(stats["proportion_ones"], 6),
            "Forensic verdict": stats["verdict"]
        })

        # Visualize LSB plane
        st.write("**LSB Bit-Plane Visualization (all channels)**")
        fig = plt.figure()
        plt.imshow(lsb * 255)  # matplotlib default colormap
        plt.axis('off')
        st.pyplot(fig)

        # Simple histogram of LSB values
        st.write("**Histogram of LSB values (0/1)**")
        fig2 = plt.figure()
        zeros = (lsb == 0).sum()
        ones = (lsb == 1).sum()
        plt.bar([0, 1], [zeros, ones])
        plt.xticks([0, 1], ["0", "1"])
        plt.xlabel("LSB value")
        plt.ylabel("Frequency")
        st.pyplot(fig2)

st.caption("Note: This demo uses simple LSB steganography and basic forensic heuristics. For unknown/advanced methods (e.g., DCT-domain), extend with ML/CNN or frequency-domain feature analysis.")
