# app.py
# Steganography Forensics — Fast LSB (MAGIC optional) + 32-bit length header + CRC32
# Run: streamlit run app.py
# Top guard will auto-relaunch with `streamlit run` if you do `python app.py`

import os
import sys
import shutil
import subprocess

# ------------------ Guard: enforce using `streamlit run` ------------------
if __name__ == "__main__" and not os.environ.get("STREAMLIT_RUN_INTERNAL"):
    streamlit_exe = shutil.which("streamlit")
    if streamlit_exe:
        print("Detected direct python run — relaunching with: streamlit run", sys.argv[0])
        env = os.environ.copy()
        env["STREAMLIT_RUN_INTERNAL"] = "1"
        subprocess.Popen([streamlit_exe, "run", sys.argv[0]], env=env)
        sys.exit(0)
    else:
        print("Please install Streamlit and run: streamlit run app.py")
        sys.exit(1)

# ------------------ Imports (normal app) ------------------
import io
import zlib
from datetime import datetime
from typing import List, Dict

import numpy as np
import streamlit as st
from PIL import Image

# Optional Torch (ML detector)
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ------------------ Config ------------------
st.set_page_config(page_title="Stego Forensics (fast) + CRC32", layout="wide")
MODEL_PATH = "models/best_stego_resnet18.pth"
DEFAULT_THRESHOLD = 70.0

# Use an optional magic header for new encodes (helps avoid false positives)
USE_MAGIC_FOR_NEW_ENCODES = True
MAGIC = b"SG1!"  # 4 bytes

# ------------------ Helpers ------------------
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    printable = sum(1 for c in b if 32 <= c <= 126)
    return printable / len(b)

def try_decodes(b: bytes):
    out = {}
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            out[enc] = b.decode(enc)
        except Exception:
            out[enc] = None
    return out

# ------------------ Encoder (LSB with length + CRC, optional magic) ------------------
def _bytes_to_bits_str(b: bytes) -> str:
    return ''.join(f"{byte:08b}" for byte in b)

def encode_with_crc(image: Image.Image, payload_bytes: bytes) -> Image.Image:
    img = to_rgb(image)
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    capacity_bits = h * w * 3

    prefix = MAGIC if USE_MAGIC_FOR_NEW_ENCODES else b""
    length = len(payload_bytes)
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    header = prefix + length.to_bytes(4, "big")
    full = header + payload_bytes + crc.to_bytes(4, "big")
    bitstream = _bytes_to_bits_str(full)

    if len(bitstream) > capacity_bits:
        raise ValueError(f"Message too long: need {len(bitstream)} bits, capacity {capacity_bits} bits")

    flat = arr.reshape(-1, 3)
    idx = 0
    # iterate over flat (fast enough) and set LSBs
    for i in range(flat.shape[0]):
        for ch in range(3):
            if idx >= len(bitstream):
                break
            flat[i, ch] = (flat[i, ch] & 0xFE) | (1 if bitstream[idx] == '1' else 0)
            idx += 1
        if idx >= len(bitstream):
            break

    stego = flat.reshape(h, w, 3).astype(np.uint8)
    return Image.fromarray(stego)

# ------------------ Fast vectorized decoder (NumPy) ------------------
def _image_lsb_bits(img: Image.Image) -> np.ndarray:
    arr = np.array(to_rgb(img), dtype=np.uint8, copy=False)
    bits = (arr & 1).reshape(-1)  # shape = (h*w*3,), values 0/1
    return bits

def _bits_to_bytes(bits: np.ndarray, nbytes: int) -> bytes:
    if nbytes <= 0:
        return b""
    upto = nbytes * 8
    trimmed = bits[:upto]
    if trimmed.size < upto:
        return b""
    packed = np.packbits(trimmed, bitorder="big")
    return packed.tobytes()[:nbytes]

def _bits_to_u32(bits: np.ndarray) -> int:
    b = _bits_to_bytes(bits, 4)
    if len(b) < 4:
        return -1
    return int.from_bytes(b, "big")

def _read_magic_and_length(all_bits: np.ndarray, expect_magic: bool):
    offset = 0
    total_bits = all_bits.size
    if expect_magic:
        if total_bits < 32:
            return False, 0, -1, "missing_magic"
        magic_bytes = _bits_to_bytes(all_bits[offset:offset+32], 4)
        if magic_bytes != MAGIC:
            return False, 0, -1, "magic_mismatch"
        offset += 32
    if total_bits < offset + 32:
        return False, 0, -1, "no_length_header"
    length = _bits_to_u32(all_bits[offset:offset+32])
    offset += 32
    return True, offset, length, None

def decode_with_crc(image: Image.Image):
    bits = _image_lsb_bits(image)
    total_bits = bits.size
    total_bytes_capacity = total_bits // 8

    def try_mode(expect_magic: bool):
        ok, offset, length, reason = _read_magic_and_length(bits, expect_magic)
        if not ok:
            return {"status": "no_payload", "reason": reason, "got_bits": int(total_bits)}
        header_bytes = (4 if expect_magic else 0) + 4  # magic? + length
        max_payload_bytes = max(0, total_bytes_capacity - header_bytes - 4)  # -CRC
        if length < 0 or length > max_payload_bytes:
            return {"status": "no_payload", "reason": "invalid_length_header", "length": int(length)}
        needed_bits = (length + 4) * 8
        remaining_bits = total_bits - offset
        if remaining_bits < needed_bits:
            got_bits = max(0, remaining_bits)
            got_bytes = got_bits // 8
            return {
                "status": "incomplete",
                "reason": "not_enough_embedded_bits",
                "length": int(length),
                "got_bits": int(got_bits),
                "got_bytes": int(got_bytes),
            }
        slice_bits = bits[offset:offset+needed_bits]
        byts = _bits_to_bytes(slice_bits, length + 4)
        if len(byts) < length + 4:
            return {
                "status": "incomplete",
                "reason": "packing_shortfall",
                "length": int(length),
                "got_bits": int(slice_bits.size),
                "got_bytes": int(len(byts)),
            }
        payload = byts[:length]
        crc_read = int.from_bytes(byts[length:length+4], "big")
        crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_read == crc_calc:
            return {
                "status": "ok",
                "length": int(length),
                "payload": payload,
                "crc": int(crc_calc),
                "got_bytes": int(len(byts)),
                "got_bits": int(needed_bits)
            }
        else:
            return {
                "status": "corrupt",
                "length": int(length),
                "payload": payload,
                "crc_read": int(crc_read),
                "crc_calc": int(crc_calc),
                "got_bytes": int(len(byts)),
                "got_bits": int(needed_bits)
            }

    res = try_mode(expect_magic=True)
    if res.get("status") != "no_payload":
        return res
    return try_mode(expect_magic=False)

# ------------------ Relaxed scanner: search LSB-bytes stream for candidates ------------------
MAX_CANDIDATE_PAYLOAD = 15000
MAX_SEARCH_PREFIX_BYTES = 4096

def _lsb_bits_as_bytes(img: Image.Image) -> bytes:
    arr = np.array(to_rgb(img), dtype=np.uint8, copy=False)
    bits = (arr & 1).reshape(-1)
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()

def _extract_payload_from_bytes_stream(lsb_bytes: bytes, offset_byte: int, expect_magic: bool) -> Dict:
    n = len(lsb_bytes)
    ptr = offset_byte
    if expect_magic:
        if ptr + 4 > n:
            return {"status":"too_short_for_magic"}
        if lsb_bytes[ptr:ptr+4] != MAGIC:
            return {"status":"magic_mismatch"}
        ptr += 4
    if ptr + 4 > n:
        return {"status":"too_short_for_length"}
    length = int.from_bytes(lsb_bytes[ptr:ptr+4], "big")
    ptr += 4
    if length < 0 or length > MAX_CANDIDATE_PAYLOAD:
        return {"status":"invalid_length", "length": length}
    if ptr + length + 4 > n:
        return {"status":"incomplete", "length": length, "available_bytes": n - ptr}
    body = lsb_bytes[ptr:ptr+length]
    crc_read = int.from_bytes(lsb_bytes[ptr+length:ptr+length+4], "big")
    crc_calc = zlib.crc32(body) & 0xFFFFFFFF
    printable = printable_ratio(body)
    decs = try_decodes(body)
    txt = decs.get("utf-8") or decs.get("latin-1") or decs.get("utf-16")
    out = {
        "status": "ok" if crc_read == crc_calc else "corrupt",
        "length": length,
        "crc_read": crc_read,
        "crc_calc": crc_calc,
        "printable_ratio": printable,
        "text_preview": txt,
        "hex_preview": body.hex()[:512],
        "offset_byte": offset_byte,
    }
    return out

def relaxed_scan_decode(img: Image.Image, max_search_prefix_bytes: int = MAX_SEARCH_PREFIX_BYTES) -> List[Dict]:
    lsb_bytes = _lsb_bits_as_bytes(img)
    n = len(lsb_bytes)
    results = []
    search_limit = min(n, max_search_prefix_bytes)
    # search for MAGIC occurrences
    idx = 0
    while True:
        found = lsb_bytes.find(MAGIC, idx, search_limit)
        if found == -1:
            break
        candidate = _extract_payload_from_bytes_stream(lsb_bytes, found, expect_magic=True)
        candidate["mode"] = "magic"
        candidate["found_at"] = found
        results.append(candidate)
        idx = found + 1
    # legacy scan for plausible length headers
    for off in range(0, max(0, search_limit - 8)):
        length = int.from_bytes(lsb_bytes[off:off+4], "big")
        if length <= 0 or length > MAX_CANDIDATE_PAYLOAD:
            continue
        if off + 4 + length + 4 > n:
            continue
        candidate = _extract_payload_from_bytes_stream(lsb_bytes, off, expect_magic=False)
        if candidate.get("status") in ("ok", "corrupt"):
            candidate["mode"] = "legacy"
            candidate["found_at"] = off
            results.append(candidate)
    # sort heuristic: prefer CRC-ok, then printable ratio high, then earlier offset
    def score(c):
        s = 0
        if c.get("status") == "ok": s += 100000
        s += int(c.get("printable_ratio", 0) * 1000)
        s -= c.get("found_at", 0) // 10
        return -s
    results = sorted(results, key=score)
    return results

# ------------------ Optional ML model utilities ------------------
if TORCH_AVAILABLE:
    @st.cache_resource
    def load_model(path: str = MODEL_PATH):
        if not os.path.exists(path):
            return None
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    PREPROC = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def predict_stego_pil(img: Image.Image, model) -> tuple:
        if model is None:
            raise RuntimeError("Model not found")
        x = PREPROC(to_rgb(img)).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            clean_p = float(probs[0] * 100.0)
            stego_p = float(probs[1] * 100.0)
            if stego_p >= clean_p:
                return "Stego", stego_p, clean_p, stego_p
            else:
                return "Clean", clean_p, clean_p, stego_p
else:
    def load_model(path: str = MODEL_PATH):
        return None
    def predict_stego_pil(img: Image.Image, model):
        raise RuntimeError("Torch not available; install torch to enable detection")

# ------------------ Session state init ------------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "stego_blob" not in st.session_state:
    st.session_state.stego_blob = None
if "model" not in st.session_state:
    st.session_state.model = None

# ------------------ Sidebar UI ------------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=MODEL_PATH)
    threshold = st.slider("Uncertainty threshold (%)", 0.0, 100.0, float(DEFAULT_THRESHOLD))
    if TORCH_AVAILABLE:
        if st.button("Load model"):
            st.session_state.model = load_model(model_path)
            st.success("Model loaded." if st.session_state.model else "Model not found.")
    else:
        st.info("Torch not available — detection disabled")

    st.markdown("---")
    st.header("Upload")
    uploaded_file = st.file_uploader("Image (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.session_state.uploaded_image = to_rgb(img)
            st.session_state.stego_blob = None
            st.success("Image loaded")
            st.image(st.session_state.uploaded_image, caption="Uploaded image", use_container_width=True)
            if uploaded_file.type in ("image/jpeg", "image/jpg"):
                st.warning("JPEG is lossy — embedded LSBs may be gone. Prefer PNG/BMP.")
        except Exception as e:
            st.error(f"Failed to read image: {e}")

    st.markdown("---")
    st.header("Session")
    if st.button("Clear chat"):
        st.session_state.chat = []
        st.success("Chat cleared")

# ------------------ Main UI ------------------
st.title("🕵️ Steganography Forensics — Fast LSB + CRC32")

col_main, col_right = st.columns([3, 1])

with col_main:
    tabs = st.tabs(["Chat", "Encode", "Decode", "Detect"])

    # Chat tab
    with tabs[0]:
        st.subheader("Chat & Operation Log")
        for e in st.session_state.chat[-200:]:
            role = e.get("role", "bot"); ts = e.get("ts", "")
            if role == "user":
                st.chat_message("user").markdown(f"**You** · <small>{ts}</small>\n\n{e['text']}", unsafe_allow_html=True)
            elif role == "system":
                st.info(f"{e.get('text')}  •  {ts}")
            else:
                st.chat_message("assistant").markdown(f"**Bot** · <small>{ts}</small>\n\n{e['text']}", unsafe_allow_html=True)

        prompt = st.chat_input('Try "encode: hello" or "decode"')
        if prompt:
            st.session_state.chat.append({"role":"user","text":prompt,"ts":now_ts()})
            if st.session_state.uploaded_image is None:
                st.session_state.chat.append({"role":"bot","text":"Upload an image first (sidebar).","ts":now_ts()})
            else:
                p = prompt.strip()
                lower = p.lower()
                if lower.startswith("encode:"):
                    msg = p.split(":", 1)[1].strip().encode("utf-8")
                    try:
                        stego = encode_with_crc(st.session_state.uploaded_image, msg)
                        buf = io.BytesIO(); stego.save(buf, format="PNG"); buf.seek(0)
                        st.session_state.stego_blob = buf.getvalue()
                        st.session_state.chat.append({"role":"bot","text":"Encoded (use Encode tab to download).","ts":now_ts()})
                    except Exception as ex:
                        st.session_state.chat.append({"role":"bot","text":f"Encoding failed: {ex}","ts":now_ts()})
                elif "decode" in lower or "extract" in lower:
                    res = decode_with_crc(st.session_state.uploaded_image)
                    if res.get("status") == "ok":
                        payload = res["payload"]
                        text = try_decodes(payload).get("utf-8") or try_decodes(payload).get("latin-1")
                        if text:
                            st.session_state.chat.append({"role":"bot","text":f"✅ CRC OK. Message:\n\n{text}","ts":now_ts()})
                        else:
                            st.session_state.chat.append({"role":"bot","text":f"✅ CRC OK (non-text {len(payload)} bytes). Hex preview shown in Decode tab.","ts":now_ts()})
                    elif res.get("status") == "corrupt":
                        st.session_state.chat.append({"role":"bot","text":f"⚠️ CRC mismatch. Declared={res.get('length')} bytes. CRC read={res.get('crc_read')} vs calc={res.get('crc_calc')}.","ts":now_ts()})
                    elif res.get("status") == "incomplete":
                        st.session_state.chat.append({"role":"bot","text":f"ℹ️ Incomplete payload: declared {res.get('length')} bytes; got {res.get('got_bytes')} bytes.","ts":now_ts()})
                    else:
                        # run relaxed scan automatically
                        candidates = relaxed_scan_decode(st.session_state.uploaded_image)
                        if candidates:
                            best = candidates[0]
                            if best["status"] == "ok":
                                preview = best["text_preview"] or best["hex_preview"]
                                st.session_state.chat.append({"role":"bot","text":f"Recovered candidate (CRC OK) mode={best['mode']} offset={best['found_at']}: {preview}","ts":now_ts()})
                            else:
                                st.session_state.chat.append({"role":"bot","text":f"Candidate found but CRC mismatch mode={best['mode']} offset={best['found_at']} (see Decode tab).","ts":now_ts()})
                        else:
                            st.session_state.chat.append({"role":"bot","text":f"No payload / invalid header ({res.get('reason','')}).","ts":now_ts()})
                else:
                    # quick detect: try direct decode then ML fallback
                    res = decode_with_crc(st.session_state.uploaded_image)
                    if res.get("status") == "ok":
                        st.session_state.chat.append({"role":"bot","text":"Direct extraction found payload (CRC OK).","ts":now_ts()})
                    else:
                        if st.session_state.model is not None:
                            try:
                                lbl, conf, cp, sp = predict_stego_pil(st.session_state.uploaded_image, st.session_state.model)
                                extra = " ⚠️ (uncertain)" if conf < threshold else ""
                                st.session_state.chat.append({"role":"bot","text":f"Model: {lbl} ({conf:.2f}%). Clean:{cp:.2f}% Stego:{sp:.2f}%{extra}","ts":now_ts()})
                            except Exception as ex:
                                st.session_state.chat.append({"role":"bot","text":f"Detection error: {ex}","ts":now_ts()})
                        else:
                            st.session_state.chat.append({"role":"bot","text":"No direct payload; model not loaded.","ts":now_ts()})

    # Encode tab
    with tabs[1]:
        st.subheader("Encode (LSB + length + CRC32)")
        if st.session_state.uploaded_image is None:
            st.info("Upload an image in the sidebar.")
        else:
            st.image(st.session_state.uploaded_image, caption="Cover preview", use_container_width=True)
            w, h = st.session_state.uploaded_image.size
            st.caption(f"Estimated capacity: {(w*h*3)//8} bytes")
            secret_text = st.text_area("Secret message (text)", height=140)
            if st.button("Encode & Download"):
                try:
                    stego_img = encode_with_crc(st.session_state.uploaded_image, secret_text.encode("utf-8"))
                    buf = io.BytesIO(); stego_img.save(buf, format="PNG"); buf.seek(0)
                    st.session_state.stego_blob = buf.getvalue()
                    st.image(stego_img, caption="Stego preview", use_container_width=True)
                    st.download_button("Download stego PNG", data=st.session_state.stego_blob, file_name="stego.png", mime="image/png")
                    st.success("Encoded successfully.")
                except Exception as ex:
                    st.error(f"Encode failed: {ex}")

    # Decode tab
    with tabs[2]:
        st.subheader("Decode & Validate")
        if st.session_state.uploaded_image is None:
            st.info("Upload an image in the sidebar.")
        else:
            st.image(st.session_state.uploaded_image, caption="Image to decode", use_container_width=True)
            if st.button("Decode & Validate"):
                res = decode_with_crc(st.session_state.uploaded_image)
                if res.get("status") == "ok":
                    pl = res["payload"]
                    decs = try_decodes(pl)
                    text = decs.get("utf-8") or decs.get("latin-1") or decs.get("utf-16")
                    st.success(f"✅ CRC OK — length {res['length']} bytes — printable ratio {printable_ratio(pl):.2f}")
                    if text:
                        st.code(text)
                    else:
                        st.code(pl.hex()[:1024])
                    st.download_button("Download payload (raw)", data=pl, file_name="extracted_payload.bin")
                elif res.get("status") == "corrupt":
                    st.warning("⚠️ Payload CRC mismatch — data may be corrupted.")
                    st.write(f"Declared length={res.get('length')}, got_bytes={res.get('got_bytes')}")
                    st.write(f"CRC read={res.get('crc_read')}, CRC calc={res.get('crc_calc')}")
                    st.code(res.get("payload", b"").hex()[:1024])
                elif res.get("status") == "incomplete":
                    st.warning("Incomplete payload: not enough embedded bits or image truncated/recompressed.")
                    st.write(f"Declared length={res.get('length')}, got_bytes={res.get('got_bytes')}")
                    # run relaxed scan automatically
                    candidates = relaxed_scan_decode(st.session_state.uploaded_image)
                    if candidates:
                        best = candidates[0]
                        st.info(f"Scanner candidate: mode={best['mode']} offset={best['found_at']} status={best['status']}")
                        if best["status"] == "ok":
                            if best["text_preview"]:
                                st.code(best["text_preview"])
                            else:
                                st.code(best["hex_preview"])
                        else:
                            st.code(best["hex_preview"])
                    else:
                        st.info("Scanner did not find plausible candidates.")
                else:
                    st.info(f"No payload found or invalid header ({res.get('reason','')}).")
                    # run relaxed scan if header invalid
                    candidates = relaxed_scan_decode(st.session_state.uploaded_image)
                    if candidates:
                        best = candidates[0]
                        st.info(f"Scanner candidate: mode={best['mode']} offset={best['found_at']} status={best['status']}")
                        if best["status"] == "ok":
                            if best["text_preview"]:
                                st.code(best["text_preview"])
                            else:
                                st.code(best["hex_preview"])
                        else:
                            st.code(best["hex_preview"])

    # Detect tab
    with tabs[3]:
        st.subheader("Detect (direct + ML fallback)")
        if st.session_state.uploaded_image is None:
            st.info("Upload an image in the sidebar.")
        else:
            st.image(st.session_state.uploaded_image, caption="Image for detection", use_container_width=True)
            quick = decode_with_crc(st.session_state.uploaded_image)
            if quick.get("status") == "ok":
                st.success("Direct extraction: payload present (CRC OK).")
                st.caption(f"Payload size: {quick['length']} bytes; printable ratio: {printable_ratio(quick['payload']):.2f}")
            elif quick.get("status") == "corrupt":
                st.warning("Direct extraction: payload present but CRC mismatch (corrupted).")
            elif quick.get("status") == "incomplete":
                st.info("Direct extraction: incomplete (not enough embedded bits).")
            else:
                st.info(f"No direct payload detected / invalid header ({quick.get('reason','')}).")
                if st.session_state.model is not None:
                    try:
                        lbl, conf, cp, sp = predict_stego_pil(st.session_state.uploaded_image, st.session_state.model)
                        st.markdown(f"**Model prediction:** {lbl} ({conf:.2f}%)")
                        st.write(f"Clean: {cp:.2f}%  |  Stego: {sp:.2f}%")
                        if lbl == "Stego" and conf >= threshold:
                            st.markdown("<h3 style='color:orange'>Likely hidden (high confidence)</h3>", unsafe_allow_html=True)
                        elif lbl == "Stego":
                            st.markdown("<h3 style='color:orange'>Possible hidden (uncertain)</h3>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h3 style='color:gray'>No hidden message detected</h3>", unsafe_allow_html=True)
                    except Exception as ex:
                        st.error(f"Detection error: {ex}")
                else:
                    st.info("ML detector not loaded.")

with col_right:
    st.subheader("Quick actions")
    if st.session_state.uploaded_image is not None:
        if st.button("Quick: Decode & Post to Chat"):
            res = decode_with_crc(st.session_state.uploaded_image)
            if res.get("status") == "ok":
                pl = res["payload"]
                decs = try_decodes(pl)
                text = decs.get("utf-8") or decs.get("latin-1")
                if text:
                    st.session_state.chat.append({"role":"bot","text":f"Quick extract ✅: {text}","ts":now_ts()})
                    st.success("Posted to chat")
                else:
                    st.session_state.chat.append({"role":"bot","text":f"Quick extract ✅ (non-text, hex): {pl.hex()[:256]}","ts":now_ts()})
                    st.success("Posted hex to chat")
            elif res.get("status") == "corrupt":
                st.session_state.chat.append({"role":"bot","text":"Quick extract ⚠️: payload present but CRC mismatch.","ts":now_ts()})
                st.warning("CRC mismatch")
            elif res.get("status") == "incomplete":
                st.warning("Quick: Incomplete payload")
            else:
                st.info(f"Quick: No payload / invalid header ({res.get('reason','')}).")

st.markdown("---")
st.caption("Tips: Use PNG/BMP (lossless). JPEG recompression destroys LSBs. Magic header helps reduce false positives.")
