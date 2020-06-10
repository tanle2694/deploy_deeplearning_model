import redis
import json
import helpers
import time
import numpy as np
import torch
from model_process import ImagenetPredict
from imagenet_labels import labels

HOST_NAME = "0.0.0.0"
PORT = 6379
REDIS_DB = 0
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 2
IMAGE_DTYPE = "float32"
IMAGE_SHAPE = (1, 3, 224, 224)
SERVER_SLEEP = 0.1


redis_db = redis.StrictRedis(host=HOST_NAME, port=6379, db=REDIS_DB)
model = ImagenetPredict()


def run_server():
    while True:
        queue = redis_db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = []
        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(data=q["image"], dtype=IMAGE_DTYPE, shape=IMAGE_SHAPE)
            batch.append(image)
            imageIDs.append(q['id'])
        if len(batch) == 0:
            time.sleep(SERVER_SLEEP)
            continue
        batch = np.concatenate(batch, axis=0)
        batch_tensor = torch.Tensor(batch, dtype=torch.float32)
        batch_tensor.to("cuda:1")
        preds = model.process_batch(batch_tensor)
        for index in len(imageIDs):
            id = imageIDs[index]
            pred = preds[index]
            redis_db.set(id, labels[pred])
        redis_db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)


if __name__ == "__main__":
    run_server()
