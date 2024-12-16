from itertools import chain
import numpy as np
from PIL import Image  # REPLACED (10 Dec)

IMAGE_SIZE = 28
IMAGE_PADDING = 0
CENTER_HEIGHT = IMAGE_SIZE - 2 * IMAGE_PADDING
CENTER_WIDTH = IMAGE_SIZE - 2 * IMAGE_PADDING


def resize_image(image, size):
    """Resize a numpy array image to the specified size using Pillow."""
    pil_image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL image
    resized_image = pil_image.resize(size, Image.BILINEAR)  # Resize with bilinear interpolation
    return np.array(resized_image) / 255.0  # Convert back to numpy array and normalize


def traces2image(traces):
    x_list, y_list = zip(*chain.from_iterable(traces))
    x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
    height, width = y_max - y_min + 1, x_max - x_min + 1
    image = np.zeros((height, width))

    for trace in traces:
        last_x, last_y = None, None
        for x, y in trace:
            if last_x is not None:
                if last_x == x:
                    yy_list = range(min(last_y, y) + 1, max(last_y, y))
                    xx_list = [x] * len(yy_list)
                else:
                    slope = (last_y - y) / float(last_x - x)
                    bias = (last_x * y - last_y * x) / float(last_x - x)
                    xx_list = range(min(last_x, x) + 1, max(last_x, x))
                    yy_list = [int(round(xx * slope + bias)) for xx in xx_list]

                for xx, yy in zip(xx_list, yy_list):
                    image[yy - y_min, xx - x_min] = 1

            image[y - y_min, x - x_min] = 1
            last_x, last_y = x, y

    if height < width:
        top_padding = (width - height) // 2
        bottom_padding = width - height - top_padding
        image = np.vstack([
            np.zeros((top_padding, width)),
            image,
            np.zeros((bottom_padding, width))
        ])
    elif width < height:
        left_padding = (height - width) // 2
        right_padding = height - width - left_padding
        image = np.hstack([
            np.zeros((height, left_padding)),
            image,
            np.zeros((height, right_padding))
        ])

    image = resize_image(image, (CENTER_HEIGHT, CENTER_WIDTH))
    image = np.vstack([
        np.zeros((IMAGE_PADDING, IMAGE_SIZE)),
        np.hstack([np.zeros((CENTER_HEIGHT, IMAGE_PADDING)), image, np.zeros((CENTER_HEIGHT, IMAGE_PADDING))]),
        np.zeros((IMAGE_PADDING, IMAGE_SIZE))
    ])

    return image.astype('float64')


if __name__ == "__main__":
    from src.backend.data_processing.prepare_data import load_symbol, DATA_SOURCE, load_ground_truth, GT_FILE
    import matplotlib.pyplot as plt

    y, symbol_set, ink_id_map = load_ground_truth(DATA_SOURCE + GT_FILE)
    for i in range(5020, 5100):  # Changed xrange to range for Python 3
        a, b = load_symbol(DATA_SOURCE + f'iso{i}.inkml')
        j = ink_id_map.get(a)
        c = traces2image(b)
        if j is not None:
            print(i, y[j])
        plt.imshow(c, cmap='gray')
        plt.show()