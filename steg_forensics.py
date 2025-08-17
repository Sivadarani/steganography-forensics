from PIL import Image
import numpy as np

def encode_image(image_path, secret_message, output_path):
    img = Image.open(image_path).convert("RGB")
    binary_msg = ''.join([format(ord(i), '08b') for i in secret_message]) + '1111111111111110'
    img_data = img.load()
    
    width, height = img.size
    msg_index = 0

    for y in range(height):
        for x in range(width):
            pixel = list(img_data[x, y])
            for channel in range(3):  # R, G, B channels
                if msg_index < len(binary_msg):
                    pixel[channel] = (pixel[channel] & ~1) | int(binary_msg[msg_index])
                    msg_index += 1
            img_data[x, y] = tuple(pixel)
            if msg_index >= len(binary_msg):
                break
        if msg_index >= len(binary_msg):
            break

    img.save(output_path)
    print(f"Message encoded and saved to: {output_path}")


def decode_image(stego_path):
    img = Image.open(stego_path)
    data = np.array(img)

    binary_data = ""
    for row in data:
        for pixel in row:
            for channel in range(3):
                binary_data += str(pixel[channel] & 1)

    # Split by 8 bits and convert to characters
    chars = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
    message = ""
    for char in chars:
        if char == '11111110':  # Delimiter detected
            break
        message += chr(int(char, 2))
    return message


def analyze_lsb_pattern(image_path):
    img = Image.open(image_path)
    data = np.array(img.convert("RGB"))

    total_pixels = data.shape[0] * data.shape[1] * 3
    ones = np.sum(data & 1)
    zeros = total_pixels - ones

    percentage = (ones / total_pixels) * 100
    return {
        "Total Pixels": total_pixels,
        "LSB 1s": int(ones),
        "LSB 0s": int(zeros),
        "Percentage of LSB 1s": f"{percentage:.2f}%"
    }


# ---------- Example Runner ----------
if __name__ == "__main__":
    original_image = "cover_image.png"  # Can be .jpg or .png
    secret = "This is a secret message!"
    encoded_output = "stego_output.png"

    encode_image(original_image, secret, encoded_output)

    print("\n🔍 Decoding Message from Stego Image...")
    decoded = decode_image(encoded_output)
    print("📩 Decoded Message:", decoded)

    print("\n🕵️‍♂️ LSB Pattern Forensics:")
    stats = analyze_lsb_pattern(encoded_output)
    for k, v in stats.items():
        print(f"{k}: {v}")
