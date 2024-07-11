import logging
import time
import os
import cv2
import camera
# 配置日志记录
from oss_utils import upload_to_oss
from model.model2 import detect_and_draw
#
request_method = None

def set_request_method(method):
    global request_method
    request_method = method

def count_persons(results):
    return results.count("person")
def run_service():
    logging.debug("Running service2")
    prev_person_count = 0
    # 确保保存图片的文件夹存在
    if not os.path.exists('../images/faces/photo'):
        os.makedirs('../images/faces/photo')
    while True:
        try:
            for frame in camera.get_camera_stream():
                #检测目标并绘制边框
                frame, result = detect_and_draw(frame)
                current_person_count = count_persons(result)

                if current_person_count > prev_person_count:
                    ##cutsave_frame_as_image
                    camera.save_frame_as_image(frame,"cut")
                    print("Alert: Number of persons increased!")

                prev_person_count = current_person_count
                cv2.imshow('Local Video Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # #模拟向服务器发送请求
                # code = "200"
                # user_id = "12345"
                # url = "jjjjj"
                # model_id = "2"q
                # request_method('service2', model_id=model_id, url=url, code=code, user_id=user_id)
                # logging.debug(f"Sent request: model_id={model_id}, url={url}, code={code}, user_id={user_id}")
                #
                # # 延时以避免过度消耗CPU资源
                # time.sleep(5)
            #     # 将本地摄像头视频流推送到 Nginx
            # push_stream_to_nginx(camera_index=0)
        except Exception as e:
            logging.error(f"Error in service2: {e}")

run_service()