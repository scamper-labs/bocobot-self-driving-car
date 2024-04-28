import os
import logging
import datetime
import pytz
import io

from fastapi import FastAPI, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.requests import Request
import uvicorn

import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()
app.count_img = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

WIDTH = 80
HEIGHT = 60
MODEL_CLASS_CONFIDENT = 0.25
UPLOAD_FOLDER = "path/to/uploads"
MODEL_PATH = "path/to/best.pt"

NGROK_PUBLISH = True
NGROK_AUTHTOKEN = "NGROK_AUTHTOKEN"
NGROK_REGION = "ap"
NGROK_PUBLISH_ONLY_HTTP = True

map_command_actions = {"left_blue": "turn left", "right_yellow": "turn right", "stop": "stop"}


def setup_logger(log_file):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def init_logger(log_dir):
    # Construct log file path with timestamp
    log_file = os.path.join(log_dir, f"logs.txt")

    # Setup logger
    logger = setup_logger(log_file)

    logger.info("Server Start!")

    return logger


def prepareImageFile(imageRaw, height=HEIGHT, width=WIDTH):
    # Read the raw image data
    dataType = np.dtype(np.uint16).newbyteorder(">")
    data = np.frombuffer(imageRaw, dtype=dataType)

    # Reshape the data into an image format (assuming width and height)
    # Replace width and height with your image dimensions
    img = data.reshape(height, width)

    # Convert from RGB565 to RGB888
    # OpenCV uses 8-bit channels for each color (0-255)
    # RGB565 has 5 bits for red, 6 bits for green, and 5 bits for blue
    # We need to expand these bits to fit 8 bits for each channel
    # You can also use bitwise operations to extract the color channels
    # and expand them manually if you prefer.
    img_plane = np.zeros((height, width, 3), dtype=np.uint8)

    img_plane[..., 2] = ((img >> 11) & 0x1F) * 255 / 31  # Red channel
    img_plane[..., 1] = ((img >> 5) & 0x3F) * 255 / 63  # Green channel
    img_plane[..., 0] = (img & 0x1F) * 255 / 31  # Blue channel

    return img_plane


def calculate_bbox_percentage(bbox, image_width, image_height):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Calculate width and height of the bounding box
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Calculate area of the bounding box
    bbox_area = bbox_width * bbox_height

    # Calculate total area of the image
    image_area = image_width * image_height

    # Calculate percentage
    percentage = bbox_area / image_area

    return percentage


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(f"{UPLOAD_FOLDER}/raw", exist_ok=True)
os.makedirs(f"{UPLOAD_FOLDER}/detect", exist_ok=True)
logger = init_logger(UPLOAD_FOLDER)

model = YOLO(MODEL_PATH)


@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def perform_healthcheck():
    logger.info("healthcheck: Everything OK!")
    return {"healthcheck": "Everything OK!"}


@app.post("/detect/")
async def detect_traffic_sign(
    image_bytes: bytes = File(...), image_format: str = "rgb", rgb_rotate_90_counterclockwise: bool = True
):

    try:
        datetime_now = datetime.datetime.now(pytz.timezone("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S.%f")[:-3]

        if image_format == "rgb":
            img_plane = prepareImageFile(image_bytes, height=HEIGHT, width=WIDTH)

            if rgb_rotate_90_counterclockwise:
                img_plane = cv2.rotate(img_plane, cv2.ROTATE_90_COUNTERCLOCKWISE)

            img = Image.fromarray(img_plane[..., ::-1])

        else:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img.save(f"{UPLOAD_FOLDER}/raw/{datetime_now}.jpg")

        # extract img height and width
        img_width, img_height = img.size

        # results = model(img, conf=MODEL_CLASS_CONFIDENT, classes=[22, 26, 27])
        results = model(img, conf=MODEL_CLASS_CONFIDENT)
        no_detect = False
        for result in results:

            if result.boxes.cls.size(dim=0) == 0:
                no_detect = True
                break

            class_id = int(result.boxes.cls[0])
            prob = float(result.boxes.conf[0]) * 100
            class_name = result.names[class_id]
            result.save(filename=f"{UPLOAD_FOLDER}/detect/{datetime_now}_{class_id}_{prob:.0f}.jpg")

            all_boxes = []
            for i, class_id in enumerate(result.boxes.cls.tolist()):
                all_boxes.append(
                    {
                        "class_name": result.names[int(class_id)],
                        "prob": float(result.boxes.conf[i]),
                        "box_pct": float(calculate_bbox_percentage(result.boxes.xyxy[i], img_width, img_height)),
                    }
                )

        if not no_detect:
            output = {
                "status": "success",
                "message": f"Got Class: {class_name}",
                "command": map_command_actions.get(class_name, "-"),
            }
            logger.info(
                f"status: success, message: {output.get('message')}, command: {output.get('command')}, raw_filename: {datetime_now}, "
                + f"all_deteced_classes: {all_boxes}"
            )

        else:
            output = {"status": "success", "message": f"No object detected", "command": "move forward"}
            logger.info(
                f"status: success, message: {output.get('message')}, command: {output.get('command')}, raw_filename: {datetime_now}"
            )

        return output

    except Exception as e:
        logger.info(f"status: error, message: {str(e)}, raw_filename: {datetime_now}")
        return {"status": "error", "message": str(e)}


@app.post("/detect_pico/")
async def detect_traffic_sign_from_pico(request: Request):

    try:
        datetime_now = datetime.datetime.now(pytz.timezone("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S.%f")[:-3]

        # Process the request here
        # Access the raw bytes of the request body
        image_bytes = await request.body()
        img_plane = prepareImageFile(image_bytes, height=HEIGHT, width=WIDTH)
        img_plane = cv2.rotate(img_plane, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = Image.fromarray(img_plane[..., ::-1])
        # img.save(f"{UPLOAD_FOLDER}/raw/{datetime_now}.jpg")

        # extract img height and width
        img_width, img_height = img.size

        # results = model(img, conf=MODEL_CLASS_CONFIDENT, classes=[22, 26, 27])
        results = model(img, conf=MODEL_CLASS_CONFIDENT)
        no_detect = False
        for result in results:

            if result.boxes.cls.size(dim=0) == 0:
                no_detect = True
                break

            class_id = int(result.boxes.cls[0])
            prob = float(result.boxes.conf[0]) * 100
            class_name = result.names[class_id]
            result.save(filename=f"{UPLOAD_FOLDER}/detect/{datetime_now}_{class_id}_{prob:.0f}.jpg")

            all_boxes = []
            for i, class_id in enumerate(result.boxes.cls.tolist()):
                all_boxes.append(
                    {
                        "class_name": result.names[int(class_id)],
                        "prob": float(result.boxes.conf[i]),
                        "box_pct": float(calculate_bbox_percentage(result.boxes.xyxy[i], img_width, img_height)),
                    }
                )

        if not no_detect:
            output = {
                "status": "success",
                "message": f"Got Class: {class_name}",
                "command": map_command_actions.get(class_name, "-"),
            }
            logger.info(
                f"status: success, message: {output.get('message')}, command: {output.get('command')}, raw_filename: {datetime_now}, "
                + f"all_deteced_classes: {all_boxes}"
            )

        else:
            output = {"status": "success", "message": f"No object detected", "command": "move forward"}
            logger.info(
                f"status: success, message: {output.get('message')}, command: {output.get('command')}, raw_filename: {datetime_now}"
            )

        return output

    except Exception as e:
        logger.info(f"status: error, message: {str(e)}, raw_filename: {datetime_now}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":

    if NGROK_PUBLISH:
        from pyngrok import ngrok, conf
        import nest_asyncio

        conf.get_default().auth_token = NGROK_AUTHTOKEN
        conf.get_default().region = NGROK_REGION

        if NGROK_PUBLISH_ONLY_HTTP:
            ngrok_tunnel = ngrok.connect(8000, proto="http", bind_tls=False)
        else:
            ngrok_tunnel = ngrok.connect(8000)

        print("Public URL:", ngrok_tunnel.public_url)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")

        nest_asyncio.apply()

    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)
