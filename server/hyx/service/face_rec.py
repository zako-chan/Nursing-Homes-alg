import face_recognition
import cv2
import os
import subprocess
import numpy as np
import threading
import time
from hyx.service.utils import camera

list_lock = threading.Lock()  # 用于线程安全地操作列表

# 数据库路径
db_path = "/home/hyx/Desktop/server/hyx/photo"

# 读取已知人脸并计算其特征
known_face_encodings = []
known_face_names = []

for file in os.listdir(db_path):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join(db_path, file)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)

        identity, person_name, person_id = file.split('-')
        known_face_names.append(f"{identity}-{person_name}-{person_id.split('.')[0]}")

# 定义存储识别名字的列表
recognized_names = []

def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) == 3:
        identity, person_name, person_id = parts
        return identity, person_name, person_id
    return None, None, None
def recognize_faces(frame,tolerance=0.4):
    # 将图像从BGR转换为RGB
    rgb_frame = frame[:, :, ::-1]

    # 查找当前帧中的所有人脸及其编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 检查检测到的人脸是否与已知人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=tolerance)
        name = "Unknown"

        # 使用第一个匹配的已知人脸的名字
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def process_frame(frame,camera_id):
    face_locations, face_names = recognize_faces(frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 绘制人脸的矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 在人脸上方绘制名字标签
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if name != "Unknown":
            with list_lock:
                if name not in recognized_names:
                    recognized_names.append(name)
                    my_custom_method(name,frame,camera_id)  # 调用自定义方法

    return frame

def clear_recognized_names():
    global recognized_names
    while True:
        time.sleep(180)  # 每180秒（3分钟）清空一次列表
        with list_lock:
            recognized_names.clear()

def my_custom_method(name,frame,camera_id):
    camera_id = int(camera_id)
    identity,name,user_id = parse_filename(name)
    user_id = int(user_id)
    path = camera.save_frame_as_image(frame, "face")
    purl = camera.upload_image_to_oss(path)
    print(purl)
    #eventInfo.face_recognition_event(user_id=user_id, image_url=purl, identity=identity, camera_id=camera_id)
    # # 自定义方法逻辑，例如发送通知或记录日志
    # print(f"Recognized and called method for: {name}")
#
def main(camera_id, pull_url, push_url):
    # 启动清空列表的线程
    clear_list_thread = threading.Thread(target=clear_recognized_names)
    clear_list_thread.daemon = True  # 设置为守护线程，在主程序结束时自动结束
    clear_list_thread.start()
    # 打开摄像头或视频流
    cap = cv2.VideoCapture(pull_url)

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # FFmpeg命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',  # 视频帧大小
        '-r', str(fps),  # 帧率
        '-i', '-',  # 从管道输入
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        push_url
    ]

    # 启动FFmpeg进程
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 如果当前帧为空，则跳过
        if frame is None:
            continue
        # # 处理当前帧
        # processed_frame = process_frame(frame,camera_id)
        processed_frame = frame

        # 将处理后的帧写入FFmpeg管道
        proc.stdin.write(processed_frame.tobytes())
        cv2.imshow('frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python face_recognition_model.py <camera_id> <pull_url> <push_url>")
        sys.exit(1)

    camera_id = sys.argv[1]
    pull_url = sys.argv[2]
    push_url = sys.argv[3]
    main(camera_id, pull_url, push_url)
# def main(camera_id, pull_url, push_url, save_path):
#     # 启动清空列表的线程
#     clear_list_thread = threading.Thread(target=clear_recognized_names)
#     clear_list_thread.daemon = True  # 设置为守护线程，在主程序结束时自动结束
#     clear_list_thread.start()
#
#     # 打开摄像头或视频流
#     cap = cv2.VideoCapture(pull_url)
#
#     # 获取视频信息
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#     # 初始化VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
#
#     # # FFmpeg命令
#     # ffmpeg_cmd = [
#     #     'ffmpeg',
#     #     '-y',  # 覆盖输出文件
#     #     '-f', 'rawvideo',
#     #     '-vcodec', 'rawvideo',
#     #     '-pix_fmt', 'bgr24',
#     #     '-s', f'{width}x{height}',  # 视频帧大小
#     #     '-r', str(fps),  # 帧率
#     #     '-i', '-',  # 从管道输入
#     #     '-c:v', 'libx264',
#     #     '-pix_fmt', 'yuv420p',
#     #     '-f', 'flv',
#     #     push_url
#     # ]
#
#     # # 启动FFmpeg进程
#     # proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 如果当前帧为空，则跳过
#         if frame is None:
#             continue
#
#         # 处理当前帧
#         processed_frame = process_frame(frame, camera_id)
#
#         # # 将处理后的帧写入FFmpeg管道
#         # proc.stdin.write(processed_frame.tobytes())
#         cv2.imshow('frame', processed_frame)
#
#         # 保存当前帧到视频文件
#         out.write(processed_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # 释放摄像头资源、关闭视频文件和所有窗口
#     cap.release()
#     out.release()
#     proc.stdin.close()
#     proc.wait()
#     cv2.destroyAllWindows()
#
# # 调用main函数，传入相应的参数
# main(camera_id=1, pull_url='/home/hyx/Desktop/server/hyx/service/recyrt.mp4', push_url='rtmp://your_push_url', save_path='/home/hyx/Desktop/server/hyx/service/recyrt2.mp4')