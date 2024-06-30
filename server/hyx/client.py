import grpc
import sys
import os
import time
from multiprocessing import Process
import importlib
import logging
import config  # 导入配置文件
from concurrent import futures

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加grpc文件夹到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'grpc'))

import message_pb2 as message_pb2
import message_pb2_grpc as message_pb2_grpc

class ClientService(message_pb2_grpc.GreeterServicer):
    def Model1Request(self, request, context):
        print(f"Received Model1 request: model_id={request.model_id}, user_id={request.user_id}, username={request.username}")
        return message_pb2.Model1ResponseMessage(code="200", url="https://example.com/model1", user_id=request.user_id, username=request.username, model_id=request.model_id)

    def Model2Request(self, request, context):
        print(f"Received Model2 request: code={request.code}, model_id={request.model_id}, url={request.url}")
        return message_pb2.Model2ResponseMessage(code="200", url=request.url, user_id=request.user_id, model_id=request.model_id)

    def Model3Request(self, request, context):
        print(f"Received Model3 request: code={request.code}, model_id={request.model_id}, url={request.url}")
        return message_pb2.Model3ResponseMessage(code="200", url=request.url, model_id=request.model_id)

def serve_client():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    message_pb2_grpc.add_GreeterServicer_to_server(ClientService(), server)
    server.add_insecure_port(f'{config.CLIENT_HOST}:{config.CLIENT_PORT}')  # 使用配置文件中的地址和端口
    server.start()
    print(f"Client service started, listening on port {config.CLIENT_PORT}")
    server.wait_for_termination()

def send_request_to_server(service_name, model_id, user_id=None, username=None, url=None, code=None):
    with grpc.insecure_channel(f'{config.SERVER_HOST}:{config.SERVER_PORT}') as channel:
        stub = message_pb2_grpc.GreeterStub(channel)
        if service_name == 'service1':
            model1_request = message_pb2.Model1RequestMessage(model_id=model_id, user_id=user_id, username=username)
            response = stub.Model1Request(model1_request)
            print(f"Server Model1 response: code={response.code}, url={response.url}, user_id={response.user_id}, username={response.username}, model_id={response.model_id}")
        elif service_name == 'service2':
            model2_request = message_pb2.Model2RequestMessage(model_id=model_id, url=url, code=code, user_id=user_id)
            stub.Model2Request(model2_request)
            print(f"Sent Model2 request: model_id={model_id}, url={url}, code={code}, user_id={user_id}")
        elif service_name == 'service3':
            model3_request = message_pb2.Model3RequestMessage(model_id=model_id, url=url, code=code)
            stub.Model3Request(model3_request)
            print(f"Sent Model3 request: model_id={model_id}, url={url}, code={code}")

def run_service_in_process(service_name, service_module):
    try:
        logging.debug(f"Starting service {service_name} in a new process.")
        if hasattr(service_module, 'run_service'):
            # 传递发送请求的方法给服务模块
            if hasattr(service_module, 'set_request_method'):
                service_module.set_request_method(lambda *args, **kwargs: send_request_to_server(service_name, **kwargs))

            # 运行服务模块
            service_module.run_service()
    except Exception as e:
        logging.error(f"Error running service {service_name}: {e}")
        # 重启服务模块
        run_service_in_process(service_name, service_module)

def run_client():
    try:
        # 动态加载服务文件
        global service_modules
        service_files = [f.split('.')[0] for f in os.listdir('service') if f.endswith('.py')]
        service_modules = {name: importlib.import_module(f'service.{name}') for name in service_files}

        processes = []
        for service_name, service_module in service_modules.items():
            process = Process(target=run_service_in_process, args=(service_name, service_module))
            processes.append(process)
            process.start()

        # 启动监听服务器请求的进程
        client_service_process = Process(target=serve_client)
        client_service_process.start()
        processes.append(client_service_process)

        for process in processes:
            process.join()
    except Exception as e:
        logging.error(f"Error running client: {e}")

if __name__ == '__main__':
    run_client()
