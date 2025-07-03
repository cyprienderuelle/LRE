import numpy as np
import random

class samples:
    def __init__(self, name, function, type=-1):
        self.name = name
        self.function = function
        self.type = type
        self.variants = []

class transform1:
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, sample):
        text = sample.function
        lines = text.split("\n")

        if len(lines) <= 2:
            return sample

        first_idx, last_idx = 1, max(2, len(lines) - 1 - self.window_size)

        if last_idx <= first_idx:
            window_start = first_idx
        else:
            window_start = np.random.randint(first_idx, last_idx)

        window_end = min(len(lines), window_start + self.window_size)
        sample.function = "\n".join(lines[window_start:window_end])
        return sample

def AugmentationNegatif(sample):
    try:
        res = samples(name=sample.name, function=transform1(window_size=random.randrange(2, len(sample.function.split("\n"))))(sample).function)
        return res
    except Exception as e:
        return sample