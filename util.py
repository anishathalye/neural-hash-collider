# Copyright (c) 2021 Anish Athalye. Released under the MIT license.

import numpy as np
import onnx
from onnx_tf.backend import prepare
from PIL import Image


def load_model(path):
    onnx_model = onnx.load(path)
    model = prepare(onnx_model, training_mode=True)
    return model


def load_seed(path):
    seed = open(path, 'rb').read()[128:]
    seed = np.frombuffer(seed, dtype=np.float32)
    seed = seed.reshape([96, 128])
    return seed


def load_image(path):
    im = Image.open(path).convert('RGB')
    im = im.resize([360, 360])
    arr = np.array(im).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 360, 360])
    return arr


def save_image(arr, path):
    arr = arr.reshape([3, 360, 360]).transpose(1, 2, 0)
    arr = (arr + 1.0) * (255.0 / 2.0)
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(path)


def hash_from_hex(hex_repr):
    n = int(hex_repr, 16)
    h = np.zeros(96)
    for i in range(96):
        h[i] = (n >> (95 - i)) & 1
    return h


def hash_to_hex(h):
    bits = ''.join(['1' if i >= 0.5 else '0' for i in h])
    return '{:0{}x}'.format(int(bits, 2), len(bits) // 4)
