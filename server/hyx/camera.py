import cv2
import config  # 导入配置文件
import os
import subprocess
import time
import logging
from multiprocessing import Queue, Process
from oss_utils import upload_to_oss
from model.model2 import detect_and_draw
# 从本地摄像头读取数据
def get_camera_stream(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

    while True:
        is_opened, frame = cap.read()
        if is_opened:
            yield frame
        else:
            logging.error("Error: Failed to read frame from camera.")
            break

# 从云端拉流读取数据
def queue_put_cloud(q, stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Could not open stream: {stream_url}")
        return

    while True:
        is_opened, frame = cap.read()
        if is_opened:
            q.put(frame)
        else:
            print(f"Error: Failed to read frame from stream: {stream_url}")
            break

def queue_get(q, window_name='image'):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def capture_and_save_images():
    if not os.path.exists('photo'):
        os.makedirs('photo')

    for frame in get_camera_stream():
        timestamp = int(time.time())
        image_path = os.path.join('photo', f'image_{timestamp}.jpg')
        cv2.imwrite(image_path, frame)
        logging.debug(f"Saved image: {image_path}")
        # 上传到OSS并获取URL
        url = upload_to_oss(image_path)
        logging.debug(f"Uploaded image to OSS: {url}")
        time.sleep(5)

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

def push_stream_from_camera(camera_index=0, rtmp_url=config.RTMP_URL):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        return

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
        while True:
            is_opened, frame = cap.read()
            if not is_opened:
                logging.error("Error: Failed to read frame from camera.")
                break

            # 检测目标并绘制边框（如果需要）
            # frame = detect_and_draw(frame)  # 如果有目标检测，取消注释

            # 将处理后的帧写入 FFmpeg 的 stdin
            proc.stdin.write(frame.tobytes())

    except Exception as e:
        logging.error(f"Error during streaming: {e}")

    finally:
        proc.stdin.close()
        proc.wait()
        cap.release()


def read_rtmp_stream(rtmp_url=config.RTMP_URL):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print("Error: Could not open RTMP stream.")
        return

    while True:
        is_opened, frame = cap.read()
        if not is_opened:
            print("Error: Failed to read frame from RTMP stream.")
            break

        cv2.imshow('RTMP Stream', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    push_stream_from_camera()
