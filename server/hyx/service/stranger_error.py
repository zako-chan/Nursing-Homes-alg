import numpy as np
import threading
import subprocess
import sys
import cv2
import os
import logging
import threading
import time
import face_recognition
from hyx.service.utils import camera
import eventInfo
import subprocess
import sys

# 将主目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model_pack.detect_pack import detect_and_draw

# 数据库路径
list_lock = threading.Lock()  # 用于线程安全地操作列表
db_path = "photo"
db_path_2 = "know"

# 读取已知人脸并计算其特征
known_face_encodings = []
known_face_names = []
# 读取已知人脸并计算其特征
known_face_encodings_2 = []
known_face_names_2 = []
# 初始化跟踪器字典和ID计数器
trackers = {}
track_id_count = 0


def load_known_faces(known_faces_dir="know"):
    global known_face_encodings, known_face_names
    for file_name in os.listdir(known_faces_dir):
        file_path = os.path.join(known_faces_dir, file_name)
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:  # 确保至少有一个人脸编码
                known_face_encodings_2.append(encodings[0])
                known_face_names_2.append(file_name.split(".")[0])
            else:
                print(f"No faces found in {file_path}, skipping.")


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    2 / 3,
                    txt_color,
                    thickness=1,
                    lineType=cv2.LINE_AA)

# 目标ID分配函数
def set_id(centroid, trackers):
    global track_id_count
    for id, data in trackers.items():
        prev_centroid, _, _ , _ , _ ,_= data
        if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < 50:
            return id
    track_id_count += 1
    return track_id_count

# 绘制轨迹函数
def traj(image, track_id, trackers):
    if track_id in trackers:
        _, trajectory, _ , _ , _,_ = trackers[track_id]
        for i in range(1, len(trajectory)):
            cv2.line(image, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

def recognize_faces(frame, tolerance=0.3):
    # 将图像从BGR转换为RGB
    rgb_frame = frame[:, :, ::-1]

    # 查找当前帧中的所有人脸及其编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 检查检测到的人脸是否与已知人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # 使用第一个匹配的已知人脸的名字
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def recognize_faces_unknow(frame, tolerance=0.5):
    # 将图像从BGR转换为RGB
    rgb_frame = frame[:, :, ::-1]

    # 查找当前帧中的所有人脸及其编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 检查检测到的人脸是否与已知人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings_2, face_encoding, tolerance=tolerance)
        name = "Unknown"

        # 使用第一个匹配的已知人脸的名字
        face_distances = face_recognition.face_distance(known_face_encodings_2, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names_2[best_match_index]

        face_names.append(name)

    return face_locations, face_names
#
# def check_fall(person_frame,net,action_net):
#     # 摔倒检测逻辑
#     # 假设如果检测到摔倒，返回 True，否则返回 False
#     height, width = person_frame.shape[:2]
#     print(width)
#     if height < width * 0.7 and height != 0:  # 假设摔倒时人体高度显著小于宽度
#         print("jhjhjhjhjh------------------------------------")
#         if run_demo(net,action_net,person_frame,256,True,[]):
#             print("fall-----------------------------------------------------------")
#             return True
#     return False
def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) == 3:
        identity, person_name, person_id = parts
        return identity, person_name, person_id
    return None, None, None

# def check_fall_and_handle(person_frame, net1, action_net, camera_id, face_names):
#     if check_fall(person_frame, net1, action_net):
#         print("Fall detected!")
#         person_id = 0
#         # #检查 face_names 是否为空
#         # if face_names and face_names[0]!="Unknown":
#         #     try:
#         #         identity, person_name, person_id = parse_filename(face_names[0])
#         #         person_id = int(person_id)
#         #     except Exception as e:
#         #         print(f"Error parsing face_names: {e}")
#         # else:
#         #     person_id = 0
#         #     print("No face names detected.")
#         path = camera.save_frame_as_image(person_frame, "fall")
#         # # 处理摔倒检测结果
#         purl = camera.upload_image_to_oss(path)
#         camera_id = int(camera_id)
#         eventInfo.fall_detection_event(image_url=purl, elderly_id=person_id, camera_id=camera_id)
#         return True
#     else:
#         # print("No fall detected.")
#         return False


# def process_frame(frame, camera_id, net1, action_net,face_names):
#     detect_and_handle_fall(frame, net1, action_net, camera_id, face_names)
#     return frame
#
# def detect_and_handle_fall(frame, net1, action_net, camera_id, face_names):
#         # 检测摔倒
#         check_fall_and_handle(frame, net1, action_net, camera_id, face_names)
#
#         # check_fall_and_handle(person_frame, net1, action_net, camera_id, face_names)

# def process_frame_thread(person_image, camera_id, net1, action_net, face_names, track_id):
#     global trackers
#     while True:
#         if check_fall_and_handle(person_image, net1, action_net, camera_id,face_names):
#             if trackers[track_id][3] == 0:
#                 trackers[track_id][4] = time.time()  # 开始计时
#             trackers[track_id][3] += 1
#             if trackers[track_id][3] >= 15 and (time.time() - trackers[track_id][4]) >= 3:
#                 print("Fall detected!")
#                 person_id = 0
#                 if face_names and face_names[0] != "Unknown":
#                     try:
#                         identity, person_name, person_id = parse_filename(face_names[0])
#                         person_id = int(person_id)
#                     except Exception as e:
#                         print(f"Error parsing face_names: {e}")
#                 else:
#                     person_id = 0
#                     print("No face names detected.")
#                 path = camera.save_frame_as_image(person_image, "fall")
#                 purl = camera.upload_image_to_oss(path)
#                 eventInfo.fall_detection_event(image_url=purl, elderly_id=person_id, camera_id=int(camera_id))
#                 trackers[track_id][3] = 0  # 重置计数器
#         else:
#             trackers[track_id][3] = 0  # 如果有任何帧为False，重置计数器
#         time.sleep(0.033)  # 每帧休眠一段时间


def get_latest_file_name(folder_path):
    # 获取文件夹中的所有文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
             os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        return None

    # 获取最新的文件
    latest_file = max(files, key=os.path.getctime)

    # 返回文件名
    return os.path.basename(latest_file)

def parse_filename_2(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) == 2:
        identity, person_id = parts
        return identity, person_id
    return None, None

def run_service(camera_id, pull_url, push_url):
    global track_id_count
    prev_person_count = 0
    # if camera_id == 1:
    #     pull_url = '/home/hyx/Desktop/server/hyx/service/stranger.mp4'
    # else:
    #     pull_url = '/home/hyx/Desktop/server/hyx/service/stranger2.mp4'
    camera_id = int(camera_id)
    cap = cv2.VideoCapture(pull_url)
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # # 加载摔倒检测的模型
    # print("加载摔倒检测的模型开始")
    # net = jit.load('openpose_fall/checkPoint/openpose.jit')
    # action_net = jit.load('openpose_fall/checkPoint/action.jit')
    # print("加载摔倒检测的模型结束")
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
    # 确保保存图片的文件夹存在
    if not os.path.exists('photo'):
        os.makedirs('photo')

    previous_centroid = [(0, 0)]  # 初始化上一帧的中心点

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            for file in os.listdir(db_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(db_path, file)
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)

                    identity, person_name, person_id = file.split('-')
                    known_face_names.append(f"{identity}-{person_name}-{person_id.split('.')[0]}")
            load_known_faces()
            results = detect_and_draw(frame)
            current_ids = []

            for (confidence, (startX, startY, endX, endY), centroid) in results:
                # 为当前目标设置ID
                track_id = set_id(centroid, trackers)
                current_ids.append(track_id)
                box_label(frame, (startX, startY, endX, endY), '#' + str(track_id) + ' person', (89, 161, 197))

                # 更新跟踪器
                if track_id in trackers:
                    prev_centroid, trajectory, alerted, frame_count, timer, id = trackers[track_id]
                    person_image = frame[startY:endY, startX:endX]
                    copy_image = person_image
                    face_locations, face_names = recognize_faces(person_image)
                    # print(face_names)

                    # 绘制人脸的矩形框和标签
                    for (top, right, bottom, left), name in zip(face_locations, face_names):

                        # 如果目标尚未报警，且识别为“Unknown”，则裁剪出人形区域并保存
                        if not alerted:
                            if name == "Unknown":
                                with list_lock:
                                    # print("aaaaaa")
                                    # 裁剪出人形区域
                                    path = camera.save_frame_as_image(frame, "stranger")  # 保存裁剪后的图像
                                    purl = camera.upload_image_to_oss(path)
                                    eventInfo.stranger_detection_event(image_url=purl, camera_id=camera_id)
                                person_image = frame[startY:endY, startX:endX]
                                face_locations, face_names = recognize_faces_unknow(copy_image)
                                # print(face_names)
                                # print("hahahahahhahha")
                                if face_names:
                                    if face_names[0] == "Unknown":
                                        #cut
                                        # print("kkkkk")
                                        refined_person_image = person_image[top:bottom, left:right]
                                        file = get_latest_file_name("know")
                                        a,b = parse_filename_2(file)
                                        b = int(b) + 1
                                        cv2.putText(person_image, str(b), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                    (0, 255, 0), 2)
                                        camera.save_frame_as_image(refined_person_image,
                                                                              "know",f'{a}-{b}.jpg')  # 保存裁剪后的图像
                                        trackers[track_id][5] = str(b)
                                        eventInfo.stranger_detection_event(image_url=purl, camera_id=camera_id,stranger_id=b)

                                    else:
                                        # print("hhhhhhh")
                                        a,b = parse_filename_2(face_names[0])
                                        b = int(b)
                                        cv2.putText(person_image, str(b), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                    (0, 255, 0), 2)
                                        trackers[track_id][5] = str(b)
                                        eventInfo.stranger_detection_event(image_url=purl, camera_id=camera_id,stranger_id=b)
                                else:
                                    # cut
                                    # print("ccccc")
                                    refined_person_image = person_image[top:bottom, left:right]
                                    file = get_latest_file_name("know")
                                    a, b = parse_filename_2(file)
                                    b = int(b) + 1
                                    cv2.putText(person_image, str(b), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                (0, 255, 0), 2)
                                    camera.save_frame_as_image(refined_person_image,
                                                               "know", f'{a}-{b}.jpg')  # 保存裁剪后的图像
                                    trackers[track_id][5] = str(b)
                                    eventInfo.stranger_detection_event(image_url=purl, camera_id=camera_id,stranger_id=b)
                                trackers[track_id][2] = True  # 标记为已报警
                            else:
                                cv2.putText(person_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (0, 255, 0), 2)
                        if trackers[track_id][5] != 0:
                            cv2.putText(person_image, trackers[track_id][5], (left+110, top -15), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                        (0, 255, 0), 2)
                        # person_image = frame[startY:endY, startX:endX]
                        # # 创建并启动线程来处理摔倒检测
                        # thread = threading.Thread(target=process_frame_thread,
                        #                           args=(person_image, camera_id, net, action_net, face_names, track_id))
                        # thread.start()

                    # 更新跟踪器的状态
                    trackers[track_id][0] = centroid
                    trackers[track_id][1].append(centroid)
                else:
                    # 初始化新的跟踪器，包含alerted标志、计时器和计数器
                    trackers[track_id] = [centroid, [centroid], False, 0,
                                          0,0]  # centroid, trajectory, alerted, frame_count, timer

                # 绘制轨迹
                traj(frame, track_id, trackers)

            # 删除该帧内没有出现的ID
            for track_id in list(trackers.keys()):
                if track_id not in current_ids:
                    del trackers[track_id]

            cv2.imshow('frame', frame)
            proc.stdin.write(frame.tobytes())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logging.error(f"Error in service: {e}")

    cap.release()
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python face_recognition_model.py <camera_id> <pull_url> <push_url>")
        sys.exit(1)

    camera_id = sys.argv[1]
    pull_url = sys.argv[2]
    push_url = sys.argv[3]
    run_service(camera_id, pull_url, push_url)