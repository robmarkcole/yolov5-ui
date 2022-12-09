# rgb(red, green, blue)
RED = (255, 0, 0)  # For objects within the ROI
GREEN = (0, 255, 0)  # For ROI box
YELLOW = (255, 255, 0)  # For objects outside the ROI

DEFAULT_IMAGE = "demo.jpg"

PRETRAINED_MODELS = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x"
]

IMAGE_SIZES = [320, 640, 1280]

MIN_CONF = 0.1
MAX_CONF = 1.0
DEFAULT_CONF = 0.5