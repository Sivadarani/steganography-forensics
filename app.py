# app.py  -- Chat UI with probabilities, threshold, encode/decode in chat
import io
from datetime import datetime
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

# -------------------- Config --------------------
st.set_page_config(page_title="Stego Chat", layout="wide")
MODEL_PATH = "models/best_stego_resnet18.pth"
DELIM = "1111111111111110"
THRESHOLD = 70.0  # percent; below this the model is considered 'uncertain'

# -------------------- LSB utils --------------------
def message_to_bits(message: str) -> str:
    return ''.join(format(ord(c), '08b') for c in message)

def bits_to_message(bits: str) -> str:
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    message = ""
    for c in chars:
        if len(c) == 8:
            message += chr(int(c, 2))
    return message

def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")

def encode_lsb(image: Image.Image, message: str) -> Image.Image:
    img = to_rgb(image)
    data = np.array(img, dtype=np.int32)
    h, w, _ = data.shape
    capacity_bits = h * w * 3
    payload = message_to_bits(message) + DELIM
    if len(payload) > capacity_bits:
        raise ValueError("Message too long for this image.")
    idx = 0
    flat = data.reshape(-1, 3)
    for i in range(flat.shape[0]):
        for ch in range(3):
            if idx >= len(payload): break
            bit = int(payload[idx])
            flat[i, ch] = (flat[i, ch] & ~1) | bit
            flat[i, ch] = max(0, min(255, flat[i, ch]))
            idx += 1
        if idx >= len(payload): break
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

# -------------------- Model utils --------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()
preproc = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def predict_stego_pil(img: Image.Image):
    """Return: label, conf (max%), clean_p, stego_p  (all percentages)"""
    if model is None:
        raise RuntimeError("Model file not found. Place your .pth at: " + MODEL_PATH)
    x = preproc(to_rgb(img)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        clean_p = float(probs[0] * 100.0)
        stego_p = float(probs[1] * 100.0)
        if stego_p >= clean_p:
            label = "Stego"
            conf = stego_p
        else:
            label = "Clean"
            conf = clean_p
        return label, conf, clean_p, stego_p

# -------------------- Session state init --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, text, ts, optional_blob, optional_fname)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "last_image_sig" not in st.session_state:
    st.session_state.last_image_sig = None
if "cached_label" not in st.session_state:
    st.session_state.cached_label = None
    st.session_state.cached_conf = None
if "cached_clean_p" not in st.session_state:
    st.session_state.cached_clean_p = None
if "cached_stego_p" not in st.session_state:
    st.session_state.cached_stego_p = None

# -------------------- Layout --------------------
col_chat, col_side = st.columns([2.2, 1])

with col_chat:
    st.markdown("<h2 style='margin:0'>🗨️ Stego Chat</h2>", unsafe_allow_html=True)
    chat_container = st.container()
    # show chat messages
    with chat_container:
        for entry in st.session_state.chat:
            role, text, ts = entry[0], entry[1], entry[2]
            blob = entry[3] if len(entry) > 3 else None
            fname = entry[4] if len(entry) > 4 else None
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You** · <small>{ts}</small>\n\n{text}", unsafe_allow_html=True)
            else:
                if blob is not None:
                    with st.chat_message("assistant"):
                        st.markdown(f"**Bot** · <small>{ts}</small>\n\n{text}", unsafe_allow_html=True)
                        st.image(Image.open(io.BytesIO(blob)), caption=fname, use_column_width=True)
                        st.download_button("Download image", data=io.BytesIO(blob), file_name=fname, mime="image/png")
                else:
                    with st.chat_message("assistant"):
                        st.markdown(f"**Bot** · <small>{ts}</small>\n\n{text}", unsafe_allow_html=True)

    # chat input area
    prompt = st.chat_input("Type a question (e.g. 'Is there hidden data?') or commands like 'decode' / 'encode: <message>'")
    if prompt:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat.append(("user", prompt, ts))
        # require image uploaded
        if st.session_state.uploaded_image is None:
            bot_reply = "Please upload an image on the right first, then ask or use the controls."
            st.session_state.chat.append(("bot", bot_reply, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            st.experimental_rerun()

        # compute detection once per image (cache)
        if st.session_state.cached_label is None:
            try:
                label, conf, clean_p, stego_p = predict_stego_pil(st.session_state.uploaded_image)
                st.session_state.cached_label = label
                st.session_state.cached_conf = conf
                st.session_state.cached_clean_p = clean_p
                st.session_state.cached_stego_p = stego_p
            except Exception as e:
                st.session_state.chat.append(("bot", f"Detection error: {e}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                st.experimental_rerun()

        lower = prompt.lower().strip()
        # simple command parsing
        if lower.startswith("encode:"):
            # user provided encode message inline
            msg_to_hide = prompt.split(":", 1)[1].strip()
            if not msg_to_hide:
                bot_text = "No message provided after `encode:`. Use the Encode box on the right or 'encode: <message>'."
                st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            else:
                try:
                    stego_img = encode_lsb(st.session_state.uploaded_image, msg_to_hide)
                    buf = io.BytesIO(); stego_img.save(buf, format="PNG"); blob = buf.getvalue()
                    fname = "stego.png"
                    bot_text = "Here is the encoded stego image."
                    st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), blob, fname))
                except Exception as e:
                    st.session_state.chat.append(("bot", f"Encoding failed: {e}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        elif "decode" in lower or "extract" in lower:
            try:
                secret = decode_lsb(st.session_state.uploaded_image)
                if secret.strip():
                    bot_text = f"Hidden message extracted:\n\n```\n{secret}\n```"
                else:
                    bot_text = "No hidden message found."
                st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            except Exception as e:
                st.session_state.chat.append(("bot", f"Decoding failed: {e}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        else:
            # default: show detection with both probabilities and threshold warning
            label = st.session_state.cached_label
            conf = st.session_state.cached_conf
            cp = st.session_state.cached_clean_p
            sp = st.session_state.cached_stego_p
            extra = ""
            if conf < THRESHOLD:
                extra = f"\n\n⚠️ *Uncertain (confidence < {THRESHOLD:.0f}%). Consider manual review or another image.*"
            bot_text = (
                f"**Prediction:** {label} ({conf:.2f}%)\n\n"
                f"**Class probabilities:** Clean: {cp:.2f}%  |  Stego: {sp:.2f}%"
                f"{extra}"
            )
            st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        st.rerun()

with col_side:
    st.markdown("<h3 style='margin:0'>📷 Image & Controls</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = img
            # reset cache on new image
            sig = (img.size, img.mode, hash(img.tobytes()[:2000]))
            if sig != st.session_state.last_image_sig:
                st.session_state.last_image_sig = sig
                st.session_state.cached_label = None
                st.session_state.cached_conf = None
                st.session_state.cached_clean_p = None
                st.session_state.cached_stego_p = None
                # optional: leave chat or add a small system note
                st.session_state.chat.append(("bot", "New image uploaded — cached detection cleared.", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        except Exception as e:
            st.error("Failed to read image: " + str(e))
            st.session_state.uploaded_image = None

    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, use_column_width=True, caption="Uploaded image")
        st.write("")

        # Manual encode flow
        st.markdown("**Encode (LSB)**")
        encode_msg = st.text_area("Enter secret message to hide (will post the stego image into chat):", key="encode_msg", height=80)
        if st.button("🔒 Encode and post to chat"):
            if not encode_msg.strip():
                st.warning("Enter a non-empty message to encode.")
            else:
                try:
                    stego_img = encode_lsb(st.session_state.uploaded_image, encode_msg)
                    buf = io.BytesIO(); stego_img.save(buf, format="PNG"); blob = buf.getvalue()
                    fname = "stego.png"
                    st.session_state.chat.append(("bot", "Here is your stego image.", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), blob, fname))
                    st.success("Encoded and posted to chat.")
                except Exception as e:
                    st.error("Encoding failed: " + str(e))

        st.markdown("---")
        # Manual decode
        if st.button("🔓 Decode and post to chat"):
            try:
                secret = decode_lsb(st.session_state.uploaded_image)
                if secret.strip():
                    bot_text = f"Hidden message extracted:\n\n```\n{secret}\n```"
                else:
                    bot_text = "No hidden message found."
                st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            except Exception as e:
                st.session_state.chat.append(("bot", "Decoding failed: " + str(e), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        st.markdown("---")
        # Recompute detection and post a detailed message + progress bars (visual)
        if st.button("🕵️ Recompute detection and post to chat"):
            if model is None:
                st.error("Model not found. Put .pth at: " + MODEL_PATH)
            else:
                try:
                    label, conf, cp, sp = predict_stego_pil(st.session_state.uploaded_image)
                    st.session_state.cached_label = label
                    st.session_state.cached_conf = conf
                    st.session_state.cached_clean_p = cp
                    st.session_state.cached_stego_p = sp
                    extra = ""
                    if conf < THRESHOLD:
                        extra = f"\n\n⚠️ *Uncertain (confidence < {THRESHOLD:.0f}%).*"
                    bot_text = (
                        f"**Recomputed detection:** {label} ({conf:.2f}%)\n\n"
                        f"Clean: {cp:.2f}%  |  Stego: {sp:.2f}%{extra}"
                    )
                    st.session_state.chat.append(("bot", bot_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    st.success("Detection posted to chat.")
                except Exception as e:
                    st.error("Detection failed: " + str(e))
    else:
        st.info("Upload an image on the right to enable Encode / Decode / Detect controls.")

# -------------------- Footer --------------------
st.markdown("<hr style='opacity:0.2'> <small>Tip: Use chat for quick questions (e.g., 'Is there hidden data?'). Use the right column to manually Encode/Decode and post results into the chat.</small>", unsafe_allow_html=True)
