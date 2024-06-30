import cv2
import numpy as np
import config

# 加载模型
net = cv2.dnn.readNetFromCaffe(config.MODEL_PROTOTXT, config.MODEL_WEIGHTS)

# 定义类别名称和对应颜色
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def detect_and_draw(frame):
    # 获取帧的宽度和高度
    (h, w) = frame.shape[:2]

    # 构建blob，并通过模型进行前向传递
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # 处理检测结果
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 过滤低置信度的检测结果
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 绘制边框和类别标签
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    return frame
