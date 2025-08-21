# generate_stego_dataset.py
# Usage:
#   python generate_stego_dataset.py
#   python generate_stego_dataset.py --message "my secret" --src dataset/train/clean --dst dataset/train/stego
import os
import argparse
from pathlib import Path
from PIL import Image
import sys

def ensure_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# ---------- Option A: try to reuse your app's encode function ----------
def try_import_encode():
    """
    Tries to import `encode_image(image_path, out_path, message)` from app.py
    or from src/encode.py if present. If found, returns the function.
    Otherwise returns None.
    """
    candidates = ["app", "src.encode", "encode", "steghide_encode"]
    for mod in candidates:
        try:
            module = __import__(mod, fromlist=["*"])
            # common function names you might have used
            for fn in ("encode_image", "encode", "steg_encode", "hide_message"):
                if hasattr(module, fn):
                    return getattr(module, fn)
        except Exception:
            continue
    return None

# ---------- Option B: simple LSB encoder fallback ----------
def lsb_encode_image(input_path, out_path, message="HELLO"):
    """
    Simple LSB text steganography for 24-bit PNGs.
    Not robust — for dataset generation only.
    """
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    max_bytes = (w * h * 3) // 8
    data = message.encode('utf-8')
    if len(data) + 1 > max_bytes:
        raise ValueError(f"Message too large for image {input_path}. Max bytes: {max_bytes}")

    # build payload: length byte(s) + data
    payload = bytes([len(data)]) + data  # one-byte length prefix (works if message <256 bytes)
    bits = []
    for byte in payload:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    # pad remaining pixels with zeros
    pixels = list(img.getdata())
    new_pixels = []
    bit_idx = 0
    total_bits = len(bits)
    for px in pixels:
        r, g, b = px
        if bit_idx < total_bits:
            r = (r & ~1) | bits[bit_idx]; bit_idx += 1
        if bit_idx < total_bits:
            g = (g & ~1) | bits[bit_idx]; bit_idx += 1
        if bit_idx < total_bits:
            b = (b & ~1) | bits[bit_idx]; bit_idx += 1
        new_pixels.append((r,g,b))
    stego = Image.new("RGB", img.size)
    stego.putdata(new_pixels)
    stego.save(out_path, "PNG")

# ---------- Main dataset generator ----------
def process_folder(src_folder, dst_folder, encode_fn, message):
    src = Path(src_folder)
    dst = Path(dst_folder)
    if not src.exists():
        print(f"Source folder {src} does not exist — skipping.")
        return
    ensure_dirs(dst)
    files = sorted([p for p in src.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    print(f"Found {len(files)} images in {src}")
    for f in files:
        out_name = dst / f.name
        try:
            if encode_fn is not None:
                # the encode_fn should accept (input_path, output_path, message) OR (PIL.Image, message)->PIL.Image
                try:
                    # try path based call first
                    encode_fn(str(f), str(out_name), message)
                except TypeError:
                    # maybe the function expects PIL input and returns PIL output
                    img = Image.open(f)
                    out_img = encode_fn(img, message)
                    if isinstance(out_img, Image.Image):
                        out_img.save(out_name)
                    else:
                        # fallback: assume returned path
                        print("encode_fn returned unexpected type; skipping", f)
                print(f"Saved (via app encode) -> {out_name}")
            else:
                lsb_encode_image(str(f), str(out_name), message)
                print(f"Saved (via LSB) -> {out_name}")
        except Exception as e:
            print(f"Error processing {f}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="dataset/train/clean", help="source clean images folder")
    parser.add_argument("--dst", type=str, default="dataset/train/stego", help="destination stego images folder")
    parser.add_argument("--val-src", type=str, default="dataset/val/clean", help="validation source clean folder")
    parser.add_argument("--val-dst", type=str, default="dataset/val/stego", help="validation destination stego folder")
    parser.add_argument("--message", type=str, default="hidden_message", help="message to hide in images")
    args = parser.parse_args()

    # try to reuse your app's encode function
    encode_fn = try_import_encode()
    if encode_fn:
        print("Using encode function imported from your app.")
    else:
        print("No encode function found in app — falling back to internal LSB encoder.")

    # process train and val
    process_folder(args.src, args.dst, encode_fn, args.message)
    process_folder(args.val_src, args.val_dst, encode_fn, args.message)
    print("Done.")

if __name__ == "__main__":
    main()
