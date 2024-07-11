import grpc
from concurrent import futures
import time
import threading
import config  # 导入配置文件
import message_pb2 as message_pb2
import message_pb2_grpc as message_pb2_grpc

class ServerService(message_pb2_grpc.GreeterServicer):
    def Model2Request(self, request, context):
        print(f"Received Model2 request: code={request.code}, model_id={request.model_id}, url={request.url}")
        return message_pb2.Model2ResponseMessage(code="200", url=request.url, user_id=request.user_id, model_id=request.model_id)

    def Model3Request(self, request, context):
        print(f"Received Model3 request: code={request.code}, model_id={request.model_id}, url={request.url}")
        return message_pb2.Model3ResponseMessage(code="200", url=request.url, model_id=request.model_id)

def send_periodic_request():
    time.sleep(10)  # 等待客户端服务启动
    with grpc.insecure_channel(f'{config.CLIENT_HOST}:{config.CLIENT_PORT}') as channel:  # 使用配置文件中的地址和端口
        stub = message_pb2_grpc.GreeterStub(channel)
        while True:
            try:
                model1_request = message_pb2.Model1RequestMessage(model_id="1", user_id="123", username="user1")
                response = stub.Model1Request(model1_request)
                print(f"Sent periodic Model1 request: code={response.code}, url={response.url}, user_id={response.user_id}, username={response.username}, model_id={response.model_id}")
            except Exception as e:
                print(f"Failed to send request: {e}")
            time.sleep(5)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_GreeterServicer_to_server(ServerService(), server)
    server.add_insecure_port(f'{config.SERVER_HOST}:{config.SERVER_PORT}')
    server.start()
    print(f"Server started, listening on port {config.SERVER_PORT}")

    # 启动定期发送请求的线程
    threading.Thread(target=send_periodic_request).start()

    server.wait_for_termination()

if __name__ == '__main__':
    serve()
