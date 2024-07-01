import grpc
import event_pb2
import event_pb2_grpc
import configparser

# 读取配置文件
def get_server_address():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['GRPC']['server_address']

# 通用函数发起请求
def send_request(request_type, request_message):
    with grpc.insecure_channel(get_server_address()) as channel:
        stub = event_pb2_grpc.EventServiceStub(channel)
        method = getattr(stub, request_type)
        return method(request_message)

# 具体请求函数
def emotion_detection_event(elderly_id, image_url):
    request = event_pb2.EmotionDetectionEventRequest(elderlyId=elderly_id, imageUrl=image_url)
    return send_request('EmotionDetectionEvent', request)

def volunteer_interaction_event(elderly_id, volunteer_id, image_url):
    request = event_pb2.VolunteerInteractionEventRequest(elderlyId=elderly_id, volunteerId=volunteer_id, imageUrl=image_url)
    return send_request('VolunteerInteractionEvent', request)

def stranger_detection_event(image_url):
    request = event_pb2.StrangerDetectionEventRequest(imageUrl=image_url)
    return send_request('StrangerDetectionEvent', request)

def fall_detection_event(image_url):
    request = event_pb2.FallDetectionEventRequest(imageUrl=image_url)
    return send_request('FallDetectionEvent', request)

def forbidden_area_invasion_detection_event(image_url):
    request = event_pb2.ForbiddenAreaInvasionDetectionEventRequest(imageUrl=image_url)
    return send_request('ForbiddenAreaInvasionDetectionEvent', request)

# 示例调用
if __name__ == '__main__':
    response = stranger_detection_event(image_url='http://example.com/image.jpg')
    print(response.message, response.code, response.data)
