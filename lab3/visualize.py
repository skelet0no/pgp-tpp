import colorsys
from pathlib import Path
import struct
import subprocess
import tempfile

from PIL import Image


def select_samples(image, class_count):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.imshow(image)
    samples = []

    for j in range(class_count):
        axes.set_title(
            f"Класс {j}: выберите пиксели и нажмите Enter\n"
            "Левая кнопка — добавить, правая — удалить последнюю точку"
        )
        points = plt.ginput(n=-1, timeout=0)
        points = [(int(x + 0.5), int(y + 0.5)) for x, y in points]
        points = [(x, y) for x, y in points if 0 <= x < image.width and 0 <= y < image.height]
        if not points:
            plt.close(figure)
            raise ValueError("Для каждого класса нужно выбрать хотя бы один пиксель")
        samples.append(points)

    plt.close(figure)
    return samples


def main():
    input_path = Path(input("Путь к изображению: ").strip()).expanduser()
    class_count = int(input("Количество классов (1–32): "))
    if not 1 <= class_count <= 32:
        raise ValueError("Количество классов должно быть от 1 до 32")

    with Image.open(input_path) as source:
        image = source.convert("RGBA")
    width, height = image.size
    samples = select_samples(image, class_count)

    directory = Path(__file__).resolve().parent
    binary = directory / "lab3"
    if not binary.is_file():
        subprocess.run(["make", "-C", str(directory)], check=True)

    output_path = input_path.with_name(input_path.stem + "_classes.png")
    with tempfile.TemporaryDirectory(prefix="lab3-") as temporary:
        input_data = Path(temporary) / "input.data"
        output_data = Path(temporary) / "output.data"
        with input_data.open("wb") as file:
            file.write(struct.pack("<ii", width, height))
            file.write(image.tobytes())

        instructions = f"{input_data}\n{output_data}\n{class_count}\n"
        for points in samples:
            instructions += f"{len(points)} " + " ".join(f"{x} {y}" for x, y in points) + "\n"
        subprocess.run([str(binary)], input=instructions, text=True, check=True)

        with output_data.open("rb") as file:
            result_width, result_height = struct.unpack("<ii", file.read(8))
            result = Image.frombytes("RGBA", (result_width, result_height), file.read())

        visualization = result.getchannel("A").convert("P")
        palette = []
        for j in range(256):
            color = colorsys.hsv_to_rgb((0.60 + j * 0.61803398875) % 1.0, 0.70, 0.95)
            palette.extend(round(channel * 255) for channel in color)
        visualization.putpalette(palette)
        visualization.convert("RGB").save(output_path)

    print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, ImportError) as error:
        raise SystemExit(f"Ошибка: {error}")
