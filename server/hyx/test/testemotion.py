import face_recognition
import cv2
import os
import numpy as np
import os
import cv2
import time
import threading
import importlib.util
from keras.preprocessing.image import img_to_array
from queue import Queue
from collections import deque

# 初始化全局变量
happy_people_list = []
list_lock = threading.Lock()  # 用于线程安全地操作列表

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
    module_path = os.path.join(os.path.dirname(__file__), '../resource', 'expression.py')

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

def experssion(frame,x,y,w,h,model):
    # 扩大裁剪区域的边界
    model.load_weights('resource/weight/cnn3_best_weights.h5')
    face_img = frame[y:y + h, x:x + w]
    # 预处理人脸图像
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)  # 增加批次维度
    face_img = face_img / 255.0

    # # 使用模型进行预测
    preds = model.predict(face_img)
    emotion_index = np.argmax(preds)
    emotion = index2emotion(emotion_index)

    # 绘制矩形框和情感标签
    frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, (255, 255, 255), 0.8)
    return frame,emotion

def process_frame(frame,model):

    face_locations, face_names = recognize_faces(frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        frame,emotion = experssion(frame,left,top,right-left,bottom-top,model)
        print(emotion)
        if emotion:  # 确保情感存在
            if name != "Unknown":
                with list_lock:
                    if name not in happy_people_list and emotion == "Happy":
                        happy_people_list.append(name)
                        print(f"好啊: {name}")
        # 绘制人脸的矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # 在人脸上方绘制名字标签
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame



# 启动清空列表的线程
clear_list_thread = threading.Thread(target=clear_happy_people_list)
clear_list_thread.daemon = True  # 设置为守护线程，在主程序结束时自动结束
clear_list_thread.start()
# 打开摄像头
cap = cv2.VideoCapture(0)
# 加载 openpose_fall 模块
expression = load_hopenet_module()

# 加载预训练的 CNN3 模型
model = expression.CNN3()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 处理当前帧
    processed_frame = process_frame(frame,model)

    # 显示处理后的帧
    cv2.imshow('Face Recognition', processed_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
