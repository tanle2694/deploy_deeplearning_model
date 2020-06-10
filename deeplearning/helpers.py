import numpy as np
import base64
import sys


def base64_decode_image(data, dtype, shape):
    data = bytes(data, encoding="utf-8")
    data = np.frombuffer(base64.decodestring(data), dtype=dtype)
    data = data.reshape(shape)
    return data