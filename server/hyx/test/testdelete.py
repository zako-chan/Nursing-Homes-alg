import face_recognition
import cv2
import os
import numpy as np
import time
import threading
import importlib.util
import subprocess
from keras.preprocessing.image import img_to_array
from queue import Queue
from collections import deque
import camera
import eventInfo
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
from model.model2 import detect_and_draw

# 初始化全局变量
happy_people_list = []
list_lock = threading.Lock()  # 用于线程安全地操作列表
pair_time_dict = {}  # 存储每对人保持在要求的距离内的时间戳


# 清空列表的线程函数
def clear_happy_people_list():
    global happy_people_list
    while True:
        time.sleep(3600)  # 每3600秒（1小时）清空一次列表
        with list_lock:
            happy_people_list.clear()


# 数据库路径
db_path = "../images/faces/photo"  # 替换为实际的数据库路径

# 读取已知人脸并计算其特征
known_face_encodings = []
known_face_names = []

# 映射情感标签
emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def index2emotion(index, lang='en'):
    if lang == 'en':
        return emotion_dict.get(index, "Unknown")
    else:
        raise NotImplementedError("Only English is supported for now")


def cv2_img_add_text(img, text, x, y, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text, (x, y), font, size, color, 2, cv2.LINE_AA)
    return img


def generate_faces(face_img, img_size=48):
    face_img = face_img / 255.0
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = [face_img]

    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img, 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    resized_images = [cv2.resize(img, (img_size, img_size)) for img in resized_images]
    resized_images = [np.expand_dims(img, axis=-1) for img in resized_images]
    resized_images = np.array(resized_images)
    return resized_images


def load_hopenet_module():
    module_name = 'expression'
    # 获取父目录的路径
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    module_path = os.path.join(parent_dir, '../resource', 'expression.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    expression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expression)
    return expression


for file in os.listdir(db_path):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_path = os.path.join(db_path, file)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)

        identity, person_name, person_id = file.split('-')
        known_face_names.append(f"{identity}-{person_name}-{person_id.split('.')[0]}")


def recognize_faces(frame):
    # 将图像从BGR转换为RGB
    rgb_frame = frame[:, :, ::-1]

    # 查找当前帧中的所有人脸及其编码
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 检查检测到的人脸是否与已知人脸匹配
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # 使用第一个匹配的已知人脸的名字
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names


def experssion(frame, x, y, w, h, model):
    model.load_weights('resource/weight/cnn3_best_weights.h5')
    face_img = frame[y:y + h, x:x + w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)  # 增加批次维度
    face_img = face_img / 255.0

    preds = model.predict(face_img)
    emotion_index = np.argmax(preds)
    emotion = index2emotion(emotion_index)

    frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, (255, 255, 255), 0.8)
    return frame, emotion


def process_frame(frame, model):
    face_locations, face_names = recognize_faces(frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        frame, emotion = experssion(frame, left, top, right - left, bottom - top, model)
        print(emotion)
        if emotion:  # 确保情感存在
            if name != "Unknown":
                with list_lock:
                    if name not in happy_people_list and emotion == "Happy":
                        happy_people_list.append(name)
                        print(f"好啊: {name}")
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


def check_pairs_and_trigger_event(results):
    global pair_time_dict
    current_time = time.time()

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            (cX1, cY1) = results[i][2]
            (cX2, cY2) = results[j][2]
            distance = np.sqrt((cX2 - cX1) ** 2 + (cY2 - cY1) ** 2)

            if distance < DISTANCE_THRESHOLD:
                name1 = results[i][1]
                name2 = results[j][1]
                pair = tuple(sorted((name1, name2)))

                if pair not in pair_time_dict:
                    pair_time_dict[pair] = current_time
                elif current_time - pair_time_dict[pair] > 2:  # 检查是否保持30秒
                    if "Happy" in [results[i][3], results[j][3]]:  # 检查是否有一个人表现开心
                        trigger_event(pair)
                        del pair_time_dict[pair]
            else:
                pair_time_dict.pop(tuple(sorted((name1, name2))), None)


def trigger_event(pair):
    print("sjkdjksj=---------------------------------------------")
    print(f"Event triggered for pair: {pair}")
    # 这里可以添加触发事件的代码，例如保存图片、发送通知等


# 启动清空列表的线程
clear_list_thread = threading.Thread(target=clear_happy_people_list)
clear_list_thread.daemon = True  # 设置为守护线程，在主程序结束时自动结束
clear_list_thread.start()

# YOLO模型文件路径
YOLO_CONFIG_PATH = '../models/yolo-coco/yolov3.cfg'
YOLO_WEIGHTS_PATH = '../models/yolo-coco/yolov3.weights'
YOLO_NAMES_PATH = '../models/yolo-coco/coco.names'

# 加载类别名称
LABELS = open(YOLO_NAMES_PATH).read().strip().split("\n")

# 加载YOLO模型
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# 非最大抑制和最小置信度阈值
NMS_THRESH = 0.3
MIN_CONF = 0.5

# 距离阈值（像素）
DISTANCE_THRESHOLD = 500


def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i], None)
            results.append(r)

    return results


def main(camera_id, pull_url, push_url):
    pull_url = 0
    print(push_url)
    cap = cv2.VideoCapture(pull_url)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if width == 0 or height == 0:
        print("Error: Could not get video frame size")
        return

    if fps == 0:
        print("Warning: Could not get FPS, defaulting to 30")
        fps = 30

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        push_url
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    expression = load_hopenet_module()
    model = expression.CNN3()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将处理后的帧写入FFmpeg管道
        results = detect_and_draw(frame)
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

            face_locations, face_names = recognize_faces(frame[startY:endY, startX:endX])
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                frame, emotion = experssion(frame, startX + left, startY + top, right - left, bottom - top, model)
                print(emotion)
                if emotion:  # 确保情感存在
                    results[i] = (prob, bbox, centroid, emotion)
                    if name != "Unknown":
                        with list_lock:
                            if name not in happy_people_list and emotion == "Happy":
                                happy_people_list.append(name)
                                print(f"好啊: {name}")
                cv2.rectangle(frame, (startX + left, startY + top), (startX + right, startY + bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (startX + left, startY + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        check_pairs_and_trigger_event(results)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                (cX1, cY1) = results[i][2]
                (cX2, cY2) = results[j][2]
                distance = np.sqrt((cX2 - cX1) ** 2 + (cY2 - cY1) ** 2)

                if distance < DISTANCE_THRESHOLD:
                    cv2.line(frame, (cX1, cY1), (cX2, cY2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{distance:.2f}", (cX1, cY1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.line(frame, (cX1, cY1), (cX2, cY2), (255, 0, 0), 1)
                    cv2.putText(frame, f"{distance:.2f}", (cX1, cY1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow('Local Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        proc.stdin.write(frame.tobytes())



    cap.release()
    proc.stdin.close()
    proc.wait()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        print("Usage: python face_recognition_model.py <camera_id> <pull_url> <push_url>")
        sys.exit(1)

    camera_id = sys.argv[1]
    pull_url = sys.argv[2]
    push_url = sys.argv[3]
    main(int(camera_id), pull_url, push_url)
