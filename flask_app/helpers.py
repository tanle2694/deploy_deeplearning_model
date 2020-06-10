import base64


def base64_encode_image(input_numpy):
    return base64.b64encode(input_numpy).decode("utf-8")