import logging
import time

request_method = None

def set_request_method(method):
    global request_method
    request_method = method

def run_service():
    logging.debug("Running service1")

    # 如果是由服务器触发的请求，则处理并返回结果
    while True:
        try:
            time.sleep(1)  # 保持进程运行，等待服务器请求
        except Exception as e:
            logging.error(f"Error in service1: {e}")
