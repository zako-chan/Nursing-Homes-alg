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
# # 将 my_folder 添加到 sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'resource'))
# # 现在可以导入 my_module
# import openpose_fall
from facenet_pytorch import MTCNN
from deepface import DeepFace


# 动态加载 openpose_fall 模块
def load_hopenet_module():
    module_name = 'openpose_fall'
    module_path = os.path.join(os.path.dirname(__file__), '../..', 'resource', 'openpose_fall.py')

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    hopenet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hopenet)
    return hopenet


# 定义图像预处理变换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_image(frame,dir,model):
    # 加载 openpose_fall 模块
    #openpose_fall = load_hopenet_module()
    face_img = frame
    direction = ""
    # 加载预训练的 Hopenet 模型
    #model_pack = openpose_fall.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # model_pack.load_state_dict(torch.load('resource/weight/hopenet_alpha2.pkl', map_location=torch.device('cpu')))
    # model_pack.eval()
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    if len(faces) == 0:
        return frame, face_img, direction

    # 假设只有一张人脸，处理第一张人脸
    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    # 扩大裁剪区域的边界
    x1 = max(0, x - 10)
    y1 = max(0, y - 10)
    w1 = min(frame.shape[1] - x, w + 20)
    h1 = min(frame.shape[0] - y, h + 20)
    face_img =frame[y1:y1 + h1, x1:x1 + w1]
    face = frame[y1:y1 + h1, x1:x1 + w1]

    # # 预处理人脸图像
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = transform(face_img).unsqueeze(0)
    #
    # # 使用 Hopenet 模型进行姿态估计
    yaw, pitch, roll = model(face_img)
    #yaw = torch.randn(1, 66)

    # 检查和调整尺寸
    if yaw.shape[1] != 198:
        print(f"Adjusting the range size to match yaw tensor's second dimension: {yaw.shape[1]}")
        arange_tensor = torch.arange(-yaw.shape[1] // 2, yaw.shape[1] // 2, dtype=torch.float32)
    else:
        arange_tensor = torch.arange(-99, 99, dtype=torch.float32)

    # if roll.shape[1] != 198:
    #     print(f"Adjusting the range size to match yaw tensor's second dimension: {yaw.shape[1]}")
    #     arange_tensor1 = torch.arange(-roll.shape[1] // 2, roll.shape[1] // 2, dtype=torch.float32)
    # else:
    #     arange_tensor1 = torch.arange(-99, 99, dtype=torch.float32)
    #
    # if pitch.shape[1] != 198:
    #     print(f"Adjusting the range size to match yaw tensor's second dimension: {yaw.shape[1]}")
    #     arange_tensor2 = torch.arange(-pitch.shape[1] // 2, pitch.shape[1] // 2, dtype=torch.float32)
    # else:
    #     arange_tensor2 = torch.arange(-99, 99, dtype=torch.float32)

    # 计算预测的 yaw
    yaw_predicted = torch.sum(torch.softmax(yaw, dim=1) * arange_tensor).item()
    # print(yaw_predicted)
    # pitch_predicted = torch.sum(torch.softmax(pitch, dim=1) * arange_tensor2).item()
    # roll_predicted = torch.sum(torch.softmax(roll, dim=1) * arange_tensor1).item()

    # 根据yaw角度判断头部方向
    if yaw_predicted < -2:
        direction = "Left"
    elif yaw_predicted > 2:
        direction = "Right"
    else:
        direction = "Front"

    # 在图像上绘制姿态信息
    font_scale = 1.2  # 调整字体大小
    if dir == direction:
        # 绘制人脸矩形框
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        cv2.putText(frame, f"right,please keep it!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    else:
        # 绘制人脸矩形框
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        cv2.putText(frame, f"please see the middle!", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), 2)

    return frame, face, direction

#this is the picture cutting
def crop_face_mtcnn(image_path):
    # 加载 MTCNN 人脸检测器
    mtcnn = MTCNN(keep_all=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # 转换为 RGB 图像
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 检测人脸
    boxes, _ = mtcnn.detect(rgb_image)

    if boxes is None:
        print("No face detected.")
        return None

    # 假设只有一张人脸，处理第一张人脸
    x1, y1, x2, y2 = map(int, boxes[0])

    # 裁剪人脸区域
    face_img = image[y1:y2, x1:x2]

    return face_img


## this is the picture FQA
def evaluate_image_quality(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image frame")

        # 初始化dlib的面部检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor_path = "resource/shape_predictor_68_face_landmarks.dat"  # 请确保该文件路径正确
    predictor = dlib.shape_predictor(predictor_path)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测面部
    faces = detector(gray)
    if len(faces) == 0:
        return 0  # 没有检测到人脸，返回0质量分数

    # 对每个检测到的面部进行质量评估
    quality_scores = []
    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in shape.parts()]

        # 计算每个面部特征点的拉普拉斯变换方差
        for (x, y) in landmarks:
            if y - 1 >= 0 and y + 2 <= gray.shape[0] and x - 1 >= 0 and x + 2 <= gray.shape[1]:
                patch = gray[y - 1:y + 2, x - 1:x + 2]
                laplacian_var = cv2.Laplacian(patch, cv2.CV_64F).var()
                quality_scores.append(laplacian_var)

    if not quality_scores:
        return 0  # 如果没有有效的拉普拉斯变换结果，返回0

    return np.mean(quality_scores)  # 返回平均质量分数
