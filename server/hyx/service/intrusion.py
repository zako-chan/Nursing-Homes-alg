import logging
import os
import cv2
import face_recognition
import threading
from hyx.service.utils import camera
import eventInfo
import subprocess
import numpy as np
import sys
# 将主目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from model_pack.detect_pack import detect_and_draw

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)

list_lock = threading.Lock()  # 用于线程安全地操作列表
recognized_names = []

# 数据库路径
db_path = "photo"  # 替换为实际的数据库路径

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

request_method = None


def set_request_method(method):
    global request_method
    request_method = method


def count_persons(results):
    return results.count("person")


def is_not_worker(name):
    return "worker" not in name


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

# 目标ID分配函数
def set_id(centroid, trackers, track_id_count):
    for id, data in trackers.items():
        prev_centroid, _,_ = data
        if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < 1000:
            return id
    track_id_count += 1
    return track_id_count

# 绘制轨迹函数
def traj(image, track_id, trackers):
    if track_id in trackers:
        _, trajectory,_ = trackers[track_id]
        for i in range(1, len(trajectory)):
            cv2.line(image, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)


# def my_custom_method(name, frame, camera_id):
#     camera_id = int(camera_id)
#     identity, name, user_id = parse_filename(name)
#     user_id = int(user_id)
#     path = camera.save_frame_as_image(frame, "cut")
#     purl = camera.upload_image_to_oss(path)
#     eventInfo.face_recognition_event(user_id=user_id, image_url=purl, identity=identity, camera_id=camera_id)
#     print(f"Recognized and called method for: {name}")
# 定义跟踪器字典和ID计数器
trackers = {}
track_id_count = 0

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
def process_cut(camera, frame, camera_id):
    path = camera.save_frame_as_image(frame, "intrusion")
    purl = camera.upload_image_to_oss(path)
    camera_id = int(camera_id)
    eventInfo.forbidden_area_invasion_detection_event(image_url=purl, camera_id=camera_id)

def run_service(camera_id, pull_url, push_url):
    prev_person_count = 0
    pull_url = '/home/hyx/Desktop/server/hyx/service/walk.mp4'
    cap = cv2.VideoCapture(pull_url)
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #     # 初始化VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = '/home/hyx/Desktop/server/hyx/service/intrusion.mp4'
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

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
    middle_line_x = width // 2

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # 绘制中间竖线
            cv2.line(frame, (middle_line_x, 0), (middle_line_x, height), (0, 255, 255), 2)

            results = detect_and_draw(frame)
            current_ids = []

            for (confidence, (startX, startY, endX, endY), centroid) in results:
                # 为当前目标设置ID
                track_id = set_id(centroid, trackers, track_id_count)
                current_ids.append(track_id)

                box_label(frame, (startX, startY, endX, endY), '#' + str(track_id) + ' person', (89, 161, 197))

                # 更新跟踪器
                if track_id in trackers:
                    prev_centroid, trajectory, alert_triggered = trackers[track_id]
                    # print(centroid[0])
                    # print(middle_line_x)
                    # print(prev_centroid[0])
                    if centroid[0] > prev_centroid[0]+10 > middle_line_x and not alert_triggered and centroid[0] - middle_line_x <100:
                        # print("hahahah")
                        face_locations, face_names = recognize_faces(frame)
                        if face_names:
                            for face_name in face_names:
                                if 'employee' not in face_name:
                                    # 创建并启动线程
                                    camera_id = int(camera_id)
                                    thread = threading.Thread(target=process_cut, args=(camera, frame, camera_id))
                                    thread.start()
                                    path = camera.save_frame_as_image(frame, "intrusion")
                                    purl = camera.upload_image_to_oss(path)

                                    eventInfo.forbidden_area_invasion_detection_event(image_url=purl,
                                                                                      camera_id=camera_id)
                                    alert_triggered = True  # 标记为已触发
                        else:
                            # print("aaaaa")
                            camera_id = int(camera_id)
                            thread = threading.Thread(target=process_cut, args=(camera, frame, camera_id))
                            thread.start()
                            path = camera.save_frame_as_image(frame, "intrusion")
                            purl = camera.upload_image_to_oss(path)
                            camera_id = int(camera_id)
                            eventInfo.forbidden_area_invasion_detection_event(image_url=purl,
                                                                              camera_id=camera_id)
                            eventInfo.stranger_detection_event(image_url=purl, camera_id=camera_id, stranger_id=0)
                            alert_triggered = True  # 标记为已触发
                    trackers[track_id] = (centroid, trajectory + [centroid], alert_triggered)
                else:
                    trackers[track_id] = (centroid, [centroid], False)

                # 绘制轨迹
                traj(frame, track_id, trackers)

            # 删除该帧内没有出现的ID
            for track_id in list(trackers.keys()):
                if track_id not in current_ids:
                    del trackers[track_id]

            cv2.imshow('frame', frame)
            proc.stdin.write(frame.tobytes())
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            logging.error(f"Error in service2: {e}")

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
    run_service(camera_id, pull_url, push_url)
