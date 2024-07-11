import os
import cv2
import numpy as np
from warnings import warn
from time import sleep
import argparse

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError
import imageio

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--video_path', default='', type=str, help='Path to the input video.')
parser.add_argument('--output_path', default='', type=str, help='Root path for output.')
parser.add_argument('--log_file', default='./pretreatment.log', type=str,
                    help='Log file path. Default: ./pretreatment.log')
parser.add_argument('--log', default=False, type=boolean_string,
                    help='If set as True, all logs will be saved. Otherwise, only warnings and errors will be saved. Default: False')
parser.add_argument('--worker_num', default=1, type=int,
                    help='How many subprocesses to use for data pretreatment. Default: 1')
parser.add_argument('--num_frames', default=100, type=int,
                    help='Number of frames to extract from the video. Default: 100')
opt = parser.parse_args()

VIDEO_PATH = opt.video_path
OUTPUT_PATH = opt.output_path
IF_LOG = opt.log
LOG_PATH = opt.log_file
WORKERS = opt.worker_num
NUM_FRAMES = opt.num_frames

T_H = 64
T_W = 64


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')


def cut_img(img, seq_info, frame_name, pid):
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % ('-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None

    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]

    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)

    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % ('-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def process_video(video_path, output_path, num_frames, pid):
    log_print(pid, START, f"Processing video {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(frame_count // num_frames, 1)

    frame_list = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frame_list) >= num_frames:
            break
        if frame_num % interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_list.append(gray_frame)
        frame_num += 1

    cap.release()

    seq_name = os.path.splitext(os.path.basename(video_path))[0]
    seq_info = [seq_name]
    out_dir = os.path.join(output_path, seq_name)
    os.makedirs(out_dir, exist_ok=True)

    count_frame = 0
    for idx, frame in enumerate(frame_list):
        img = cut_img(frame, seq_info, f"frame_{idx:04d}", pid)
        if img is not None:
            save_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            imageio.imwrite(save_path, img)
            count_frame += 1

    log_print(pid, FINISH, f'Processed {count_frame} frames. Saved to {out_dir}')

    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % ('-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)


def main():
    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('Pretreatment Start.\n'
          'Video path: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (VIDEO_PATH, OUTPUT_PATH, LOG_PATH, WORKERS))

    video_files = [VIDEO_PATH]  # You can modify this to process multiple videos
    for video_file in video_files:
        results.append(pool.apply_async(process_video, args=(video_file, OUTPUT_PATH, NUM_FRAMES, pid)))
        sleep(0.02)
        pid += 1

    pool.close()
    unfinish = 1
    while unfinish > 0:
        unfinish = 0
        for i, res in enumerate(results):
            try:
                res.get(timeout=0.1)
            except Exception as e:
                if type(e) == MP_TimeoutError:
                    unfinish += 1
                    continue
                else:
                    print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                          i, type(e))
                    raise e
    pool.join()


if __name__ == "__main__":
    main()
