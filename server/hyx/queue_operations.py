import cv2

def queue_put(q, video_name="/path/to/video.mp4"):
    cap = cv2.VideoCapture(video_name)
    while True:
        is_opened, frame = cap.read()
        if is_opened:
            q.put(frame)
        else:
            break

def queue_get(q, window_name='image'):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_NORMAL)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
