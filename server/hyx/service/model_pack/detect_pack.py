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
PERSON_IDX = CLASSES.index("person")  # 找到person的索引

def detect_and_draw(frame):
    (H, W) = frame.shape[:2]
    results = []

    # 构建blob，并通过模型进行前向传递
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == PERSON_IDX:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # 计算中心点
                centroidX = (startX + endX) // 2
                centroidY = (startY + endY) // 2
                centroid = (centroidX, centroidY)

                result = (confidence, (startX, startY, endX, endY), centroid)
                results.append(result)

                # 绘制边框和类别标签
                label = "Person: {:.2f}%".format(confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results


