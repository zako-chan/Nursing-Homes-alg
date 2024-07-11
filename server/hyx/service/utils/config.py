SERVER_HOST = 'localhost'
SERVER_PORT = 9999

CLIENT_HOST = 'localhost'
CLIENT_PORT = 8503
# 定义多个流的URL
STREAM_URLS = [
    "http://8.130.148.5:8080/live/hyxtest.flv",
    "http://path.to/your/stream2",
    "http://path.to/your/stream3",
    "http://path.to/your/stream4",
    "http://path.to/your/stream5"
]


# 阿里云OSS配置
OSS_ENDPOINT = '**********************'
OSS_ACCESS_KEY_ID = '*********************'
OSS_ACCESS_KEY_SECRET = '*********************'
OSS_BUCKET_NAME = 'hyxzjbnb'

##
MODEL_PROTOTXT = 'resource/deploy.prototxt'
MODEL_WEIGHTS = 'resource/mobilenet_iter_73000.caffemodel'

# RTMP 推流地址
RTMP_URL = 'rtmp://8.130.148.5/live/hyxtest2'


DEEPSORT_MODEL = "/home/hyx/Desktop/server/hyx/service/weights/ckpt.t7"
GAITSET_MODEL = "/home/hyx/Desktop/server/hyx/service/weights/gaitset.ptm"

# 模型文件路径
MODEL_PROTOTXT = '/home/hyx/Desktop/server/hyx/resource/deploy.prototxt'
MODEL_WEIGHTS = '/home/hyx/Desktop/server/hyx/resource/mobilenet_iter_73000.caffemodel'