import logging
import time
import sys
import os
import cv2
import config  # 导入配置文件
import os
import subprocess
import time
import logging
import camera
from multiprocessing import Queue, Process
from oss_utils import upload_to_oss
from model.model1 import process_image , evaluate_image_quality
request_method = None
import threading
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import sys
import os
import dlib
import importlib.util
from facenet_pytorch import MTCNN
from deepface import DeepFace

# 添加grpc文件夹到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'grpc'))

import message_pb2 as message_pb2
import message_pb2_grpc as message_pb2_grpc

def set_request_method(method):
    global request_method
    request_method = method


# 动态加载 openpose_fall 模块
def load_hopenet_module():
    module_name = 'openpose_fall'
    module_path = os.path.join(os.path.dirname(__file__), '', '../resource', 'openpose_fall.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    hopenet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hopenet)
    return hopenet

def background_task(username, user_id, identity, url, camera_index):
    global stop_thread, stop_thread_event
    while not stop_thread.is_set():
        rtmp_url = camera_index
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logging.error("Error: Could not open camera.")
            stop_thread_event.set()
            return
        hopenet = load_hopenet_module()
        # 加载预训练的 Hopenet 模型
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        model.load_state_dict(torch.load('resource/weight/hopenet_alpha2.pkl', map_location=torch.device('cpu')))
        model.eval()
        # 获取摄像头的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # FFmpeg 命令
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',  # 视频分辨率
            '-r', str(fps),  # 帧率
            '-i', '-',  # 从标准输入读取数据
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-f', 'flv',
            rtmp_url
        ]

        # 启动 FFmpeg 进程
        proc = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

        try:
            last_frame = None
            front_frame_start_time = None
            while not stop_thread.is_set():
                is_opened, frame = cap.read()
                if not is_opened:
                    logging.error("Error: Failed to read frame from camera.")
                    break
                frame, a, b = process_image(frame, "Front",model)
                last_picture = None
                max_score = 0
                if b == "Front":
                    score = evaluate_image_quality(a)
                    print(score)
                    if max_score < score:
                        max_score = score
                        last_picture = a
                    if front_frame_start_time is None:
                        front_frame_start_time = time.time()
                    elif time.time() - front_frame_start_time >= 3:
                        # 判断是否满足停止条件
                        if max_score > 1100:
                            print("满足停止条件，后台任务停止")
                            path = camera.save_frame_as_image(last_picture,"photo", f'{identity}-{username}-{user_id}.jpg')
                            purl = camera.upload_image_to_oss(path)
                            print(purl)
                            #store
                            message = message_pb2.Model1ResponseMessage(
                                model_id="1", url=purl, code="200", user_id=user_id, username=username,identity=identity
                            )
                            print(message)
                            cap.release()
                            stop_thread.set()
                            break

                else:
                    max_score = 0
                    last_picture = None
                    front_frame_start_time = None

                # 确保帧是有效的

                if frame is not None and frame.size != 0:
                    proc.stdin.write(frame.tobytes())
                    # 在本地显示视频流
                    cv2.imshow('Local Video Stream', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_thread.set()
                        break

        except Exception as e:
            logging.error(f"Error during streaming: {e}")

        finally:
            proc.stdin.close()
            proc.wait()
            cap.release()
            stop_thread_event.set()  # 通知主线程后台任务已完成

def some_condition_met(last_picture):
    s = evaluate_image_quality(last_picture)
    print(s)
    return s >= 70

def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) == 3:
        identity, person_name, person_id = parts
        return identity, person_name, person_id
    return None, None, None

def check_and_delete_duplicates(folder_path, identity, person_id):
    duplicate_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_identity, _, file_id = parse_filename(file)
            if file_identity == identity and file_id == person_id:
                duplicate_files.append(os.path.join(root, file))

    if duplicate_files:
        for file in duplicate_files:
            try:
                os.remove(file)
                print(f"已删除重复文件: {file}")
            except Exception as e:
                print(f"删除文件时出错: {file}, 错误: {e}")
    else:
        print("没有找到重复的文件")

def collection(username,user_id,identity,url,camera_index):
    global stop_thread, stop_thread_event
    stop_thread = threading.Event()
    stop_thread_event = threading.Event()
    url = 0
    camera_index = 'rtmp://8.130.148.5/live/hyxtest2'
    check_and_delete_duplicates("../images/faces/photo", identity, user_id)
    background_thread = threading.Thread(target=background_task, args=(username, user_id, identity, url, camera_index))
    background_thread.start()

    print("start")
    stop_thread_event.wait()  # 等待后台线程完成

    print("thread stop")
    stop_thread.set()  # 确保在collection函数结束时，后台任务也会停止
    background_thread.join()


def run_service():
    logging.debug("Running service1")

    # 如果是由服务器触发的请求，则处理并返回结果
    #request_method('service1', model_id="1", url="jkjkjkjkj", code="200", user_id="user_id", username="username")
    while True:
        try:
            time.sleep(1)  # 保持进程运行，等待服务器请求
        except Exception as e:
            logging.error(f"Error in service1: {e}")


collection("hhh","111","worker","","")