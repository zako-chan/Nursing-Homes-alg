import logging
import time

request_method = None

def set_request_method(method):
    global request_method
    request_method = method

def run_service():
    logging.debug("Running service3")

    while True:
        try:
            # 模拟向服务器发送请求
            code = "200"
            url = "https://example.com/model3"
            model_id = "3"
            request_method('service3', model_id=model_id, url=url, code=code)
            logging.debug(f"Sent request: model_id={model_id}, url={url}, code={code}")

            # 延时以避免过度消耗CPU资源
            time.sleep(5)
        except Exception as e:
            logging.error(f"Error in service3: {e}")
