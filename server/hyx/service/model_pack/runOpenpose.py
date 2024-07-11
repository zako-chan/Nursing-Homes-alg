import argparse
import cv2
import numpy as np
from torch import from_numpy, jit
from hyx.models.openpose_fall.openpose_modules.keypoints import extract_keypoints, group_keypoints
from hyx.models.openpose_fall.openpose_modules.pose import Pose
from action_detect.detect import action_detect
import os
from math import ceil, floor
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(parent_dir)
from utils.contrastImg import coincide

os.environ["PYTORCH_JIT"] = "0"


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name, code_name):
        self.file_name = file_name
        self.code_name = str(code_name)
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration

        # print(self.cap.get(7),self.cap.get(5))
        cv2.putText(img, self.code_name, (5, 35),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        return img


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(floor((min_dims[0] - h) / 2.0)))
    pad.append(int(floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape  # 实际高宽
    scale = net_input_height_size / height  # 将实际高缩放到期望高的缩放倍数

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # 缩放后的图像
    scaled_img = normalize(scaled_img, img_mean, img_scale)  # 归一化图像
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)  # 填充到高宽为stride 整数倍的值

    tensor_img = from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()  # 有HWC转成CHW(BGR格式)
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)  # 得到网络输出

    # print(stages_output)

    stage2_heatmaps = stages_output[-2]  # 最后一个stage的热图
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))  # 最后一个stage的热图作为最终的热图
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                          interpolation=cv2.INTER_CUBIC)  # 热图放大upsample_ratio倍

    stage2_pafs = stages_output[-1]  # 最后一个stage的paf
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))  # 最后一个stage的paf作为最终的paf
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                      interpolation=cv2.INTER_CUBIC)  # paf 放大upsample_ratio倍

    return heatmaps, pafs, scale, pad  # 返回热图,paf,输入模型图象相比原始图像缩放倍数,输入模型图像padding尺寸


def run_demo(net, action_net, img, height_size, cpu, boxList):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    orig_img = img.copy()
    fallFlag = 0

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        posebox = (int(pose.bbox[0]), int(pose.bbox[1]), int(pose.bbox[0]) + int(pose.bbox[2]), int(pose.bbox[1]) + int(pose.bbox[3]))
        if boxList:
            coincideValue = coincide(boxList, posebox)
            if len(pose.getKeyPoints()) >= 10 and coincideValue >= 0.2 and pose.lowerHalfFlag < 3:
                current_poses.append(pose)
        else:
            current_poses.append(pose)

    for pose in current_poses:
        pose.img_pose = pose.draw(img, is_save=True, show_draw=True)
        crown_proportion = pose.bbox[2] / pose.bbox[3]
        pose = action_detect(action_net, pose, crown_proportion)

        if pose.pose_action == 'fall':
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255), thickness=3)
            cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            fallFlag = 1
            return True
        else:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
            return False

    return False



def detect_main(video_name=''):
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                           This is just for quick results preview.
                           Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, default='openpose.jit',
                        help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+',
                        default='data/pics',
                        help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--code_name', type=str, default='None', help='the name of video')
    # parser.add_argument('--track', type=int, default=0, help='track pose id in video')
    # parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if video_name != '':
        args.code_name = video_name

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    net = jit.load('openpose_fall/checkPoint/openpose.jit')

    # *************************************************************************
    action_net = jit.load('openpose_fall/checkPoint/action.jit')
    # ************************************************************************

    if args.video != '':
        frame_provider = VideoReader(args.video, args.code_name)
    else:
        images_dir = []
        if os.path.isdir(args.images):
            for img_dir in os.listdir(args.images):
                images_dir.append(os.path.join(args.images, img_dir))
            frame_provider = ImageReader(images_dir)
        else:
            img = cv2.imread(args.images, cv2.IMREAD_COLOR)
            frame_provider = [img]

        # *************************************************************************

        # args.track = 0
    # camera = VideoReader('rtsp://admin:a1234567@10.34.131.154/cam/realmonitor?channel=1&subtype=0',args.code_name)

    run_demo(net, action_net, frame_provider, args.height_size, True, [])


if __name__ == '__main__':
    detect_main()
