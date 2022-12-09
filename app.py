import streamlit as st
import torch
from PIL import Image, ImageDraw
from typing import Tuple
import numpy as np
import const

DEFAULT_IMAGE = "assets/farm.jpg"


def draw_box(
    draw: ImageDraw,
    box: Tuple[float, float, float, float],
    text: str = "",
    color: Tuple[int, int, int] = (255, 255, 0),
) -> None:
    """
    Draw a bounding box on and image.
    """

    line_width = 3
    font_height = 8
    y_min, x_min, y_max, x_max = box
    (left, right, top, bottom) = (
        x_min,
        x_max,
        y_min,
        y_max,
    )
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=line_width,
        fill=color,
    )
    if text:
        draw.text(
            (left + line_width, abs(top - line_width - font_height)), text, fill=color
        )


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    return model


model = get_model()

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    pil_image = Image.open(img_file_buffer)

else:
    pil_image = Image.open(DEFAULT_IMAGE)

results = model(pil_image, size=640)  # reduce size=320 for faster inference
df = results.pandas().xyxy[0]

draw = ImageDraw.Draw(pil_image)
for _, obj in df.iterrows():
    name = obj["name"]
    confidence = obj["confidence"]
    box_label = f"{name}"

    draw_box(
        draw,
        (obj["ymin"], obj["xmin"], obj["ymax"], obj["xmax"]),
        text=box_label,
        color=const.RED,
    )

st.image(
    np.array(pil_image),
    caption=f"Processed image",
    use_column_width=True,
)

st.table(df)
