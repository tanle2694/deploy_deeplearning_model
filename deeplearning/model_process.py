import torch
from imagenet_labels import labels

HOST_NAME = "0.0.0.0"
PORT = 6379
REDIS_DB = 0
IMAGE_QUEUE = "image_queue"


class ImagenetPredict():
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.4.0', 'resnet50', pretrained=True)
        self.model.eval()
        self.model.to("cuda:1")

    def process_batch(self, images):
        with torch.no_grad():
            output = self.model(images)
        return output.argmax(1).cpu().data.numpy()




