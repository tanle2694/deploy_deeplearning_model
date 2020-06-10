from flask import Flask
from PIL import Image
from torchvision import transforms

import flask
import numpy as np
import redis
import uuid
import time
import json
import io
import helpers

HOST_NAME = "0.0.0.0"
PORT = 6379
REDIS_DB = 0
IMAGE_QUEUE = "image_queue"
CLIENT_SLEEP = 0.1
WAITING_TIME = 10

redis_db = redis.StrictRedis(host=HOST_NAME, port=6379, db=REDIS_DB)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    input_tensor = preprocess(image)
    return input_tensor.numpy()


app = Flask(__name__)


@app.route('/')
def hello_word():
    return "Hellow world"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image_array = prepare_image(image)
            image_array = image_array.copy(order="C")
            key = str(uuid.uuid4())
            image_str = helpers.base64_encode_image(image_array)
            data = {"id": key, "image": image_str}
            redis_db.rpush(IMAGE_QUEUE, json.dumps(data))
            start_processing = time.time()
            while True:
                if time.time() - start_processing > WAITING_TIME:
                    break
                output = redis_db.get(key)
                if output is None:
                    time.sleep(CLIENT_SLEEP)
                    continue
                output = output.decode("utf-8")
                data["predictions"] = json.loads(output)
                data["success"] = True
                redis_db.delete(key)
                break
    return flask.jsonify(data)