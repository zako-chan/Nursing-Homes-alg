import face_recognition
import cv2
import os
import numpy as np
import threading
import time
import importlib.util
from keras.preprocessing.image import img_to_array

# 初始化全局变量
happy_people_list = []
sad_people_list = []
list_lock = threading.Lock()  # 用于线程安全地操作列表

# 清空列表的线程函数
def clear_happy_people_list():
    global happy_people_list,sad_people_list
    while True:
        time.sleep(3600)  # 每3600秒（1小时）清空一次列表
        with list_lock:
            happy_people_list.clear()
            sad_people_list.clear()

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

def recognize_faces(frame,tolerance=0.3):
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

def cv2_img_add_text(img, text, x, y, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text, (x, y), font, size, color, 2, cv2.LINE_AA)
    return img

def load_hopenet_module():
    module_name = 'expression'
    # 获取父目录的路径
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    module_path = os.path.join(parent_dir, 'resource', 'expression.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    expression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expression)
    return expression

def experssion(frame, x, y, w, h, model):
    model.load_weights('./resource/weight/cnn3_best_weights.h5')
    face_img = frame[y:y + h, x:x + w]
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0
    preds = model.predict(face_img)
    emotion_index = np.argmax(preds)
    emotion = index2emotion(emotion_index)
    frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, (255, 255, 255), 0.8)
    return frame, emotion

def index2emotion(index, lang='en'):
    emotion_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    if lang == 'en':
        return emotion_dict.get(index, "Unknown")
    else:
        raise NotImplementedError("Only English is supported for now")

def parse_filename(filename):
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    parts = name.split('-')
    if len(parts) == 3:
        identity, person_name, person_id = parts
        return identity, person_name, person_id
    return None, None, None

def process_frame(frame, model,camera_id):

    face_locations, face_names = recognize_faces(frame)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        frame, emotion = experssion(frame, left, top, right - left, bottom - top, model)
        print(emotion)
        if emotion:
            if name != "Unknown":
                with list_lock:
                    # print(emotion)
                    # print(sad_people_list)
                    if name not in happy_people_list and emotion == "Happy":
                        happy_people_list.append(name)
                        print(name)
                        # camera_id = int(camera_id)
                        # identity, name, user_id = parse_filename(name)
                        # user_id = int(user_id)
                        # path = camera.save_frame_as_image(frame, "emotion")
                        # purl = camera.upload_image_to_oss(path)
                        # eventInfo.emotion_detection_event(elderly_id=user_id, image_url=purl, emotion="Happy",
                        #                                   camera_id = camera_id)
                    elif name not in sad_people_list and emotion == "Sad":
                        sad_people_list.append(name)
                        # camera_id = int(camera_id)
                        # identity, name, user_id = parse_filename(name)
                        # user_id = int(user_id)
                        # path = camera.save_frame_as_image(frame, "emotion")
                        # purl = camera.upload_image_to_oss(path)
                        # eventInfo.emotion_detection_event(elderly_id=user_id, image_url=purl, emotion="sad",
                        #                                        camera_id=camera_id)
                        # print(f"好啊: {name}")
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main(camera_id, pull_url, push_url):
    # 启动清空列表的线程
    clear_list_thread = threading.Thread(target=clear_happy_people_list)
    clear_list_thread.daemon = True  # 设置为守护线程，在主程序结束时自动结束
    clear_list_thread.start()
    # pull_url = '/home/hyx/Desktop/server/hyx/service/emotion.mp4'
    cap = cv2.VideoCapture(pull_url)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #     # 初始化VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = '/home/hyx/Desktop/server/hyx/service/emotion2.mp4'
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

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

        processed_frame = process_frame(frame, model,camera_id)
        proc.stdin.write(processed_frame.tobytes())
        cv2.imshow('frame', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    main(int(camera_id), pull_url, push_url)
