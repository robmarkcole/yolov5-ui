import streamlit as st
import torch
from PIL import Image, ImageDraw
from typing import Tuple
import numpy as np
import const
import time

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


@st.cache(allow_output_mutation=True, show_spinner=True)
def get_model(model_id : str = "yolov5s"):
    model = torch.hub.load("ultralytics/yolov5", model_id)
    return model

# Settings
st.sidebar.title("Settings")
model_id = st.sidebar.selectbox("Pretrained model", const.PRETRAINED_MODELS, index=1)
img_size = st.sidebar.selectbox("Image resize for inference", const.IMAGE_SIZES, index=1)
CONFIDENCE = st.sidebar.slider(
    "Confidence threshold",
    const.MIN_CONF,
    const.MAX_CONF,
    const.DEFAULT_CONF,
)

model = get_model(model_id)
st.title(f"{model_id}")

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    pil_image = Image.open(img_file_buffer)

else:
    pil_image = Image.open(const.DEFAULT_IMAGE)

st.text(f"Input image width and height: {pil_image.width} x {pil_image.width}")
start_time = time.time()
results = model(pil_image, size=img_size)
end_time = time.time()

df = results.pandas().xyxy[0]
df = df[df["confidence"] > CONFIDENCE]

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

st.text(f"Time to inference: {round(time.time() - end_time, 2)} sec")

st.table(df)
