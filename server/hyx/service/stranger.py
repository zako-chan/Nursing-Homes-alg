# import cv2
# import numpy as np
# import os
# import sys
# import threading
# import subprocess
# # 将主目录添加到 sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# from service.deep_sort.deep_sort import DeepSort
# from model_pack.detect_pack import detect_and_draw
# from hyx.service.gait_process_images.process_image import process_single_image
# import shutil
# from hyx.service.gait_process_images.process import main as process_main
# from hyx.service.gait_process_images.testwalk import main as walk_main
#
# def delete_directory(directory_path):
#     if os.path.exists(directory_path):
#         shutil.rmtree(directory_path)
#         # print(f"Deleted directory: {directory_path}")
#     # else:
#         # print(f"Directory does not exist: {directory_path}")
#
#
# def process_frame(frame_id, track_id, startX, startY, endX, endY, frame, track_output_dir, track_all_dir, track_cut_dir,
#                   track_info,flag):
#     if frame_id % 10 == 0 and frame_id < 100:  # 假设视频帧率为30帧/秒，90帧相当于3秒
#         # 增加边框
#         margin = 10
#         a = max(0, startX - margin - 100)
#         b = max(0, startY - margin)
#         c = min(frame.shape[1], endX + margin + 100)
#         d = min(frame.shape[0], endY + margin)
#         person_img = frame[b:d, a:c]
#
#         save_path = os.path.join(track_output_dir, f'frame_{frame_id}_id_{track_id}.jpg')
#         cv2.imwrite(save_path, person_img)
#         process_single_image(save_path, track_all_dir)
#
# def draw_detections(frame, detections, track_id, output_base_dir, cut_dir, all_dir, track_info):
#     # 创建与track_id绑定的新文件夹
#     track_output_dir = os.path.join(output_base_dir, str(track_id))
#     track_cut_dir = os.path.join(cut_dir, str(track_id))
#     track_all_dir = os.path.join(all_dir, str(track_id))
#     frame_id = track_info[track_id]["frame_id"]
#     # b = track_info[track_id]["b"]
#     if frame_id <100:
#         if not os.path.exists(track_output_dir):
#             os.makedirs(track_output_dir)
#         if not os.path.exists(track_cut_dir):
#             os.makedirs(track_cut_dir)
#         if not os.path.exists(track_all_dir):
#             os.makedirs(track_all_dir)
#
#     for (confidence, (startX, startY, endX, endY), centroid) in detections:
#         # if frame_id % 10 == 0 and frame_id < 100:  # 假设视频帧率为30帧/秒，90帧相当于3秒
#         #     # 增加边框
#         #     margin = 10
#         #     a = max(0, startX - margin - 100)
#         #     b = max(0, startY - margin)
#         #     c = min(frame.shape[1], endX + margin + 100)
#         #     d = min(frame.shape[0], endY + margin)
#         #     person_img = frame[b:d, a:c]
#         #
#         #     save_path = os.path.join(track_output_dir, f'frame_{frame_id}_id_{track_id}.jpg')
#         #     cv2.imwrite(save_path, person_img)
#         #     process_single_image(save_path, track_all_dir)
#         # if frame_id == 100:
#         #     from process import main as process_main
#         #
#         #     process_main(
#         #         image_dir=track_all_dir,
#         #         output_path=track_cut_dir,
#         #         log_file='./pretreatment.log',
#         #         log=False,
#         #         worker_num=1
#         #     )
#         #     from testwalk import main as testwalk_main
#         #     a, b = testwalk_main(track_cut_dir)
#         #     track_info[track_id]["b"] = b
#         #     if b > 0.9:
#         #         track_info[track_id]["b"] = a
#         #         delete_directory(track_output_dir)
#         #         delete_directory(track_all_dir)
#         #         delete_directory(track_cut_dir)
#         #     else :
#         #         delete_directory(track_output_dir)
#         #         delete_directory(track_all_dir)
#         # 启动一个新线程来处理帧
#         if frame_id < 50:
#             flag = 0
#             thread = threading.Thread(target=process_frame, args=(
#                 frame_id, track_id, startX, startY, endX, endY, frame, track_output_dir, track_all_dir, track_cut_dir,
#                 track_info, flag))
#             thread.start()
#
#         elif frame_id == 50:
#             process_main(
#                 image_dir=track_all_dir,
#                 output_path=track_cut_dir,
#                 log_file='gait_process_mp4/pretreatment.log',
#                 log=False,
#                 worker_num=1
#             )
#             a, b = walk_main(track_cut_dir)
#             track_info[track_id]["b"] = "Unknow"
#             track_info[track_id]["flag"] = 1
#             if b > 0.99:
#                 track_info[track_id]["b"] = a
#                 delete_directory(track_output_dir)
#                 delete_directory(track_all_dir)
#                 delete_directory(track_cut_dir)
#             else:
#                 delete_directory(track_output_dir)
#                 delete_directory(track_all_dir)
#
#
#         name = track_info[track_id]["b"]
#         if name == 0:
#             name = "Unknow"
#         cv2.putText(frame, f'{name}', (startX + 90, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         cv2.putText(frame, f'ID: {track_id}', (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#
#     track_info[track_id]["frame_id"] += 1  # 增加frame_id
#
# def capture_frames_from_video(video_path, push_url, output_dir, cut_dir, all_dir, model_path):
#
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     video_path = "/home/hyx/Desktop/server/hyx/service/strangerwlp2.mp4"
#     # print(push_url)
#     # 初始化DeepSort
#     tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, model_path=model_path)
#     cap = cv2.VideoCapture(video_path)
#     track_info = {}
#     # 获取视频信息
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     #     # 初始化VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     save_path = '/home/hyx/Desktop/server/hyx/service/stranger2.mp4'
#     out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
#     # FFmpeg命令
#     ffmpeg_cmd = [
#         'ffmpeg',
#         '-y',  # 覆盖输出文件
#         '-f', 'rawvideo',
#         '-vcodec', 'rawvideo',
#         '-pix_fmt', 'bgr24',
#         '-s', f'{width}x{height}',  # 视频帧大小
#         '-r', str(fps),  # 帧率
#         '-i', '-',  # 从管道输入
#         '-c:v', 'libx264',
#         '-pix_fmt', 'yuv420p',
#         '-f', 'flv',
#         push_url
#     ]
#
#     # 启动FFmpeg进程
#     proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 使用原始帧进行检测
#         results = detect_and_draw(frame.copy())
#
#         # 提取检测框和置信度
#         bboxes = np.array([r[1] for r in results])
#         confidences = np.array([r[0] for r in results])
#         clss = np.array([0 for _ in results])  # 所有目标都是人，类别为0
#         # 检查bbox是否为空
#         if len(bboxes) > 0:
#             # 更新跟踪器
#             outputs = tracker.update(bboxes, confidences, clss, frame)
#             for output in outputs:
#                 bbox, track_id = output[:4], output[4]
#                 if track_id not in track_info:
#                     track_info[track_id] = {"frame_id": 0, "b": 0,"flag":0}  # 初始化frame_id和b
#                 draw_detections(frame, results, track_id, output_dir, cut_dir, all_dir, track_info)
#
#         cv2.imshow("Frame", frame)
#         proc.stdin.write(frame.tobytes())
#         out.write(frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#
#     cap.release()
#     proc.stdin.close()
#     proc.wait()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: python face_recognition_model.py <camera_id> <pull_url> <push_url>")
#         sys.exit(1)
#
#     camera_id = sys.argv[1]
#     pull_url = sys.argv[2]
#     push_url = sys.argv[3]
#
#     output_dir = "/home/hyx/Desktop/server/hyx/service/in"
#     cut_dir = "/home/hyx/Desktop/server/hyx/data"
#     all_dir = "/home/hyx/Desktop/server/hyx/service/out"
#     model_path = "/home/hyx/Desktop/server/hyx/service/weights/ckpt.t7"  # DeepSort模型路径
#     capture_frames_from_video(video_path = pull_url, push_url = push_url,output_dir = output_dir, cut_dir=cut_dir,all_dir = all_dir,model_path = model_path)
