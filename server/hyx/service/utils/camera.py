import cv2
import os
import time
import logging
from multiprocessing import Queue, Process
from oss_utils import upload_to_oss


def save_frame_as_image(frame, folder='photo', file_name=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if file_name is None:
        timestamp = int(time.time())
        file_name = f'image_{timestamp}.jpg'

    image_path = os.path.join(folder, file_name)
    cv2.imwrite(image_path, frame)
    logging.debug(f"Saved image: {image_path}")
    return image_path

def upload_image_to_oss(image_path):
    if not image_path:
        logging.error("No image path provided for upload.")
        return None

    url = upload_to_oss(image_path)
    logging.debug(f"Uploaded image to OSS: {url}")
    return url
