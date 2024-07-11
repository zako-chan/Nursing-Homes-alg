import cv2
import numpy as np

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
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), 1)

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

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
