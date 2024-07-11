import subprocess
import logging
from model_pack.hopenet_pack import evaluate_image_quality
request_method = None
import threading
import time
import cv2
import os
import sys
import importlib.util

# 将主目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from . import eventInfo

def set_request_method(method):
    global request_method
    request_method = method

# 动态加载 openpose_fall 模块
def load_hopenet_module():
    module_name = 'openpose_fall'
    module_path = os.path.join(os.path.dirname(__file__), '..', 'resource', 'openpose_fall.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    hopenet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hopenet)
    return hopenet

def background_task(username, user_id, identity, url, camera_index):
    global stop_thread, stop_thread_event
    while not stop_thread.is_set():
        rtmp_url = camera_index
        url = '/home/hyx/Desktop/server/hyx/service/yrt3.mp4'
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logging.error("Error: Could not open camera.")
            stop_thread_event.set()
            return
        # openpose_fall = load_hopenet_module()
        # # 加载预训练的 Hopenet 模型
        # model_pack = openpose_fall.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        # model_pack.load_state_dict(torch.load('/home/hyx/Desktop/server/hyx/resource/weight/hopenet_alpha2.pkl', map_location=torch.device('cpu')))
        # model_pack.eval()
        # 获取摄像头的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))  # Adjust frame size (640, 480) as needed

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

        # try:
        #     last_frame = None
        #     front_frame_start_time = None
        #     while not stop_thread.is_set():
        #         is_opened, frame = cap.read()
        #         if not is_opened:
        #             logging.error("Error: Failed to read frame from camera.")
        #             try:
        #                 eventInfo.collection_event(user_id=user_id, username=username, identity=identity,
        #                                            image_url="error")
        #             except Exception as e:
        #                 logging.error(f"Error in collection_event: {e}")
        #             break
        #         # frame, a, b = process_image(frame, "Front", model_pack)
        #         # last_picture = None
        #         # max_score = 0
        #         # if b == "Front":
        #         #     score = evaluate_image_quality(a)
        #         #     print(score)
        #         #     if max_score < score:
        #         #         max_score = score
        #         #         last_picture = a
        #         #     if front_frame_start_time is None:
        #         #         front_frame_start_time = time.time()
        #         #     elif time.time() - front_frame_start_time >= 3:
        #         #         # 判断是否满足停止条件
        #         #         if max_score > 800:
        #         #             print("满足停止条件，后台任务停止")
        #         #             path = camera.save_frame_as_image(last_picture, "photo", f'{identity}-{username}-{user_id}.jpg')
        #         #             purl = camera.upload_image_to_oss(path)
        #         #             print(purl)
        #         #             try:
        #         #                 eventInfo.collection_event(user_id=user_id, username=username, identity=identity, image_url=purl)
        #         #             except Exception as e:
        #         #                 logging.error(f"Error in collection_event: {e}")
        #         #             finally:
        #         #                 cap.release()
        #         #                 stop_thread.set()
        #         #                 break
        #         #
        #         # else:
        #         #     max_score = 0
        #         #     last_picture = None
        #         #     front_frame_start_time = None
        #         purl ="https://hyxzjbnb.oss-cn-beijing.aliyuncs.com/volunteer-yrt-1.jpg"
        #         eventInfo.collection_event(user_id=user_id, username=username, identity=identity, image_url=purl)
        #         if frame is not None and frame.size != 0:
        #             proc.stdin.write(frame.tobytes())
        #
        #             cv2.imshow('Local Video Stream', frame)
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 stop_thread.set()
        #                 break
        #
        # except Exception as e:
        #     logging.error(f"Error during streaming: {e}")
        #
        # finally:
        #     proc.stdin.close()
        #     proc.wait()
        #     out.release()
        #     cap.release()
        #     stop_thread_event.set()  # 通知主线程后台任务已完成
        try:
            last_frame = None
            front_frame_start_time = None
            stop_time = time.time() + 30  # 10 seconds from now
            while not stop_thread.is_set():
                is_opened, frame = cap.read()
                if not is_opened:
                    logging.error("Error: Failed to read frame from camera.")
                    try:
                        eventInfo.collection_event(user_id=user_id, username=username, identity=identity,
                                                   image_url="error")
                    except Exception as e:
                        logging.error(f"Error in collection_event: {e}")
                    break


                if frame is not None and frame.size != 0:
                    proc.stdin.write(frame.tobytes())
                    cv2.imshow('Local Video Stream', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_thread.set()
                        break

                if time.time() >= stop_time:
                    stop_thread.set()
                    purl = "https://hyxzjbnb.oss-cn-beijing.aliyuncs.com/volunteer-yrt-1.jpg"
                    eventInfo.collection_event(user_id=user_id, username=username, identity=identity, image_url=purl)
                    cap.release()
                    cv2.destroyAllWindows()
                    break

        finally:
            stop_thread.set()
            purl = "https://hyxzjbnb.oss-cn-beijing.aliyuncs.com/volunteer-yrt-1.jpg"
            eventInfo.collection_event(user_id=user_id, username=username, identity=identity, image_url=purl)
            cap.release()
            cv2.destroyAllWindows()




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
    person_id = str(person_id)
    duplicate_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_identity,file_name,file_id = parse_filename(file)
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
    check_and_delete_duplicates("photo", identity, user_id)
    background_thread = threading.Thread(target=background_task, args=(username, user_id, identity, url, camera_index))
    background_thread.start()

    print("start")
    stop_thread_event.wait()  # 等待后台线程完成

    print("thread stop")
    stop_thread.set()  # 确保在collection函数结束时，后台任务也会停止
    background_thread.join()

def removeurl(user_id,identity,username):
    print(identity,user_id)
    check_and_delete_duplicates("photo", identity, user_id)
    print('ok')

def run_service():
    logging.debug("Running service1")

    # 如果是由服务器触发的请求，则处理并返回结果
    #request_method('service1', model_id="1", url="jkjkjkjkj", code="200", user_id="user_id", username="username")
    while True:
        try:
            time.sleep(1)  # 保持进程运行，等待服务器请求
        except Exception as e:
            logging.error(f"Error in service1: {e}")

