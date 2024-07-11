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


# parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('--image_dir', default='', type=str, help='Path to the input images directory.')
# parser.add_argument('--output_path', default='', type=str, help='Root path for output.')
# parser.add_argument('--log_file', default='./pretreatment.log', type=str,
#                     help='Log file path. Default: ./pretreatment.log')
# parser.add_argument('--log', default=False, type=boolean_string,
#                     help='If set as True, all logs will be saved. Otherwise, only warnings and errors will be saved. Default: False')
# parser.add_argument('--worker_num', default=1, type=int,
#                     help='How many subprocesses to use for data pretreatment. Default: 1')
# opt = parser.parse_args()
#
# IMAGE_DIR = opt.image_dir
# OUTPUT_PATH = opt.output_path
# IF_LOG = opt.log
# LOG_PATH = opt.log_file
# WORKERS = opt.worker_num

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


def process_image(image_path, output_path, pid):
    log_print(pid, START, f"Processing image {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        message = f"Failed to read image {image_path}"
        warn(message)
        log_print(pid, FAIL, message)
        return

    # Resize image
    img = cv2.resize(img, (T_W, T_H), interpolation=cv2.INTER_CUBIC)

    # Save the processed image
    seq_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_path, f"{seq_name}.png")
    imageio.imwrite(save_path, img)

    log_print(pid, FINISH, f'Processed and saved image to {save_path}')


def main(image_dir, output_path, log_file, log, worker_num):
    global IMAGE_DIR, OUTPUT_PATH, LOG_PATH, IF_LOG, WORKERS
    IMAGE_DIR = image_dir
    OUTPUT_PATH = output_path
    LOG_PATH = log_file
    IF_LOG = log
    WORKERS = worker_num

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    pool = Pool(WORKERS)
    results = list()
    pid = 0

    print('Pretreatment Start.\n'
          'Image directory: %s\n'
          'Output path: %s\n'
          'Log file: %s\n'
          'Worker num: %d' % (IMAGE_DIR, OUTPUT_PATH, LOG_PATH, WORKERS))

    image_files = [os.path.join(IMAGE_DIR, file) for file in os.listdir(IMAGE_DIR) if
                   file.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        results.append(pool.apply_async(process_image, args=(image_file, OUTPUT_PATH, pid)))
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


# if __name__ == "__main__":
#     main(
#         image_dir='/home/hyx/Desktop/server/hyx/service/out/image',
#         output_path='/home/hyx/Desktop/server/hyx/data/image',
#         log_file='./pretreatment.log',
#         log=False,
#         worker_num=1
#     )
