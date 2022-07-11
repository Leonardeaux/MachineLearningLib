import numpy as np
import os
import random
from PIL import Image
from typing import Tuple


def get_dataset(type: str) -> Tuple[np.array, np.array]:
    training_data_set = []
    for filename in os.listdir(f"D:/Data/{type}/BERLINE"):
        with Image.open(os.path.join(f"D:/Data/{type}/BERLINE", filename)).convert('L') as img:
            data_image = np.asarray(img)
            training_data_set.append((data_image.flatten(), np.array([-1], dtype=np.float64)))
    for filename in os.listdir(f"D:/Data/{type}/SUV"):
        with Image.open(os.path.join(f"D:/Data/{type}/SUV", filename)).convert('L') as img:
            data_image = np.asarray(img)
            training_data_set.append((data_image.flatten(), np.array([1], dtype=np.float64)))

    random.shuffle(training_data_set)

    x = []
    y = []

    for data in training_data_set:
        x.append(data[0])
        y.append(data[1])

    return np.array(x), np.array(y)