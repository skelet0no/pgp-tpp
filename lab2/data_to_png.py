from PIL import Image
import struct
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "rb") as f:
    width, height = struct.unpack("<ii", f.read(8))

    pixels = f.read(width * height * 4)

image = Image.frombytes("RGBA", (width, height), pixels)
alpha = image.getchannel("A")
alpha = alpha.point(lambda _: 255)
image.putalpha(alpha)
image.save(output_file)

print(f"{width}x{height} -> {output_file}")