import logging
import time
import os
import cv2
import camera
from camera import get_camera_stream, save_frame_as_image, upload_image_to_oss, push_stream_to_nginx
# 配置日志记录
from oss_utils import upload_to_oss
from model.model2 import detect_and_draw
request_method = None

def set_request_method(method):
    global request_method
    request_method = method

def run_service():
    logging.debug("Running service2")
    # 确保保存图片的文件夹存在
    if not os.path.exists('photo'):
        os.makedirs('photo')
    while True:
        try:
            for frame in camera.get_camera_stream():
                # 检测目标并绘制边框
                frame = detect_and_draw(frame)
                image_path = camera.save_frame_as_image(frame)
                purl = camera.upload_image_to_oss(image_path)

                # 模拟向服务器发送请求
                code = "200"
                user_id = "12345"
                url = purl
                model_id = "2"
                request_method('service2', model_id=model_id, url=url, code=code, user_id=user_id)
                logging.debug(f"Sent request: model_id={model_id}, url={url}, code={code}, user_id={user_id}")

                # 延时以避免过度消耗CPU资源
                time.sleep(5)
                # 将本地摄像头视频流推送到 Nginx
            push_stream_to_nginx(camera_index=0)
        except Exception as e:
            logging.error(f"Error in service2: {e}")
