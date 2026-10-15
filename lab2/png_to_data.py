from PIL import Image
import struct
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

image = Image.open(input_file).convert("RGBA")
width, height = image.size

with open(output_file, "wb") as f:
    f.write(struct.pack("<ii", width, height))

    for r, g, b, a in image.get_flattened_data():
        f.write(bytes([r, g, b, a]))

print(f"{width}x{height} -> {output_file}")
