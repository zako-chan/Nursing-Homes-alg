from concurrent import futures
import grpc
import vision_pb2
import vision_pb2_grpc
import subprocess
import os
import threading
from service import service1
# 假设我们有三个模型，分别对应于人脸识别、摔倒检测和入侵检测
models = {
    1: "face_rec",
    2: "emotion",
    3: "emotion_distance",
    4: "intrusion",
    5: "fall",
    6: "fire",
    7: "stranger",
    8: "cut"
}


# 模拟的进程管理器，用于管理摄像头ID和对应的进程
class ProcessManager:
    def __init__(self):
        self.processes = {}

    def start_process(self, camera_id, model, pull_url, push_url):
        if camera_id in self.processes:
            self.stop_process(camera_id)
        # 启动新的进程，并传递参数
        script_path = os.path.join("service", f"{model}.py")
        process = subprocess.Popen(["python", script_path, str(camera_id), pull_url, push_url])
        self.processes[camera_id] = process
        print(f"Started {model} for camera {camera_id} with PID {process.pid}")

    def stop_process(self, camera_id):
        if camera_id in self.processes:
            process = self.processes[camera_id]
            process.terminate()
            print(f"Stopped process for camera {camera_id} with PID {process.pid}")
            del self.processes[camera_id]


process_manager = ProcessManager()


class VisionServiceServicer(vision_pb2_grpc.VisionServiceServicer):



    def FaceCollection(self, request, context):
        response = vision_pb2.CommonResopnse(code=0, message=f'Collected face for user {request.username}')

        def collection_thread(username, user_id, identity, pull_url, push_url):
            service1.collection(username, user_id, identity, pull_url, push_url)
        # 创建并启动线程
        thread = threading.Thread(target=collection_thread, args=(
        request.username, request.user_id, request.identity, request.pull_url, request.push_url))
        thread.start()
        return response

    def RemoveUrl(self, request, context):
        # 实现RemoveUrl请求处理逻辑
        response = vision_pb2.CommonResopnse(code=0, message=f'Removed URL for user {request.username}')
        service1.removeurl(request.user_id, request.identity, request.username)
        return response

    def StartVisonService(self, request, context):
        # 实现StartVisonService请求处理逻辑
        camera_id = request.camera_id
        model_id = request.model_id
        pull_url = request.pull_url
        push_url = request.push_url

        if model_id == 0:
            process_manager.stop_process(camera_id)
            response = vision_pb2.CommonResopnse(code=0, message=f'Stopped vision service for camera {camera_id}')
        elif model_id in models:
            model = models[model_id]
            process_manager.start_process(camera_id, model, pull_url, push_url)
            response = vision_pb2.CommonResopnse(code=0, message=f'Started vision service for camera {camera_id}')
        else:
            response = vision_pb2.CommonResopnse(code=1, message=f'Unknown model_id: {model_id}')

        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vision_pb2_grpc.add_VisionServiceServicer_to_server(VisionServiceServicer(), server)
    server.add_insecure_port('[::]:8503')
    server.start()
    print("Server started, listening on port 8503.")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
