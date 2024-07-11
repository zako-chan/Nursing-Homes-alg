import sys
import time
import subprocess

from models.experimental import attempt_load
from hyx.models.utils.datasets import letterbox
from hyx.models.utils.general import non_max_suppression, scale_coords, check_img_size
from hyx.models.utils.torch_utils import select_device
import cv2
import numpy as np
import torch
from hyx.service.utils import camera
import eventInfo


class Smoke_File_Detector():
    def __init__(self):
        self.opt = {
            'weights': 'service/weights/smoke2.pt',
            'source': 'inference/images',
            'img_size': 640,
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'device': '',
            'classes': None,
            'agnostic_nms': False,
            'augment': False
        }

        self.device = select_device(self.opt['device'])

        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model_pack
        self.model = attempt_load(self.opt['weights'], map_location=self.device)
        self.imgsz = check_img_size(self.opt['img_size'], s=self.model.stride.max())
        if self.half:
            self.model.half()

        self.last_trigger_time = 0  # 初始化上次触发事件时间

    def trigger_alert(self, detection_type, confidence,camera_id,frame):
        path = camera.save_frame_as_image(frame, "fire")
        purl = camera.upload_image_to_oss(path)
        camera_id = int(camera_id)
        eventInfo.fire_detection_event(image_url=purl,
                                          camera_id=camera_id)
        #print(f"Alert: Detected {detection_type} with confidence {confidence:.2f}")

    def detect_video(self, camera_id, pull_url, push_url):
        pull_url = '/home/hyx/Desktop/server/hyx/service/video1.mp4'
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
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            im0 = frame
            img = letterbox(frame, new_shape=self.opt['img_size'])[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)  # faster

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img, augment=self.opt['augment'])[0]

            pred = non_max_suppression(pred, self.opt['conf_thres'], self.opt['iou_thres'], classes=self.opt['classes'],
                                       agnostic=self.opt['agnostic_nms'])

            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            current_time = time.time()
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det = det.cpu().numpy()
                    for *xyxy, conf, cls in det:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(im0, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        if names[int(cls)] in ['fire', 'smoke'] and conf > 0.50:
                            if current_time - self.last_trigger_time >= 10:
                                self.trigger_alert(names[int(cls)], conf,camera_id,frame)
                                self.last_trigger_time = current_time


            proc.stdin.write(frame.tobytes())
            cv2.imshow("Detected Video", im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
    det = Smoke_File_Detector()
    det.detect_video(camera_id, pull_url, push_url)
