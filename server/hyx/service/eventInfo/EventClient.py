import grpc
import sys
import os
import os
import sys
from . import event_pb2
from . import event_pb2_grpc
import configparser

# 读取配置文件
def get_server_address():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['GRPC']['server_address']

# 通用函数发起请求
def send_request(request_type, request_message):
    with grpc.insecure_channel('192.168.43.52:9999') as channel:
        stub = event_pb2_grpc.EventServiceStub(channel)
        method = getattr(stub, request_type)
        return method(request_message)

# 具体请求函数
def collection_event(user_id,identity,username,image_url):
    request = event_pb2.UpdateImageUrlRequest(user_id=user_id,identity=identity,username=username,image_url=image_url)
    print('ok')
    return send_request('UpdateImageUrl', request)

def emotion_detection_event(elderly_id, image_url,emotion,camera_id):
    request = event_pb2.EmotionDetectionEventRequest(elderlyId=elderly_id, imageUrl=image_url,emotion =emotion,cameraId=camera_id)
    return send_request('EmotionDetectionEvent', request)

def face_recognition_event(user_id, identity, image_url, camera_id):
    request = event_pb2.FaceRecognitionEventRequest(
        user_id=user_id,
        identity=identity,
        imageUrl=image_url,
        cameraId=camera_id
    )
    return send_request('FaceRecognitionEvent', request)

def volunteer_interaction_event(elderly_id, volunteer_id, image_url,camera_id):
    request = event_pb2.VolunteerInteractionEventRequest(elderlyId=elderly_id, volunteerId=volunteer_id, imageUrl=image_url,cameraId = camera_id)
    return send_request('VolunteerInteractionEvent', request)

def stranger_detection_event(image_url,camera_id,stranger_id):
    request = event_pb2.StrangerDetectionEventRequest(imageUrl=image_url,cameraId=camera_id,strangerId = stranger_id )
    return send_request('StrangerDetectionEvent', request)

def fall_detection_event(image_url,elderly_id,camera_id):
    request = event_pb2.FallDetectionEventRequest(imageUrl=image_url,elderlyId = elderly_id,cameraId=camera_id)
    return send_request('FallDetectionEvent', request)

def forbidden_area_invasion_detection_event(image_url,camera_id):
    request = event_pb2.ForbiddenAreaInvasionDetectionEventRequest(imageUrl=image_url, cameraId=camera_id)
    return send_request('ForbiddenAreaInvasionDetectionEvent', request)


def fire_detection_event(image_url,camera_id):
    request = event_pb2.FireDetectionEventRequest(imageUrl=image_url, cameraId=camera_id)
    return send_request('FireDetectionEvent', request)

# 示例调用
if __name__ == '__main__':
    response = collection_event(user_id=1,identity='identity',username='username',image_url='http://example.com/image.jpg')
    print(response.message, response.code, response.data)
