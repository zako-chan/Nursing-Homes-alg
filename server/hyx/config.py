SERVER_HOST = 'localhost'
SERVER_PORT = 50051

CLIENT_HOST = 'localhost'
CLIENT_PORT = 50052

# 定义多个流的URL
STREAM_URLS = [
    "http://path.to/your/stream1",
    "http://path.to/your/stream2",
    "http://path.to/your/stream3",
    "http://path.to/your/stream4",
    "http://path.to/your/stream5"
]


# 阿里云OSS配置
OSS_ENDPOINT = 'oss-cn-beijing.aliyuncs.com'
OSS_ACCESS_KEY_ID = 'LTAI5tGuLnH1EvkEG6XhbZqG'
OSS_ACCESS_KEY_SECRET = 'izeXdapmAcAseHyUaLfOVNHKFLrRWP'
OSS_BUCKET_NAME = 'hyxzjbnb'

##
MODEL_PROTOTXT = 'resource/deploy.prototxt'
MODEL_WEIGHTS = 'resource/mobilenet_iter_73000.caffemodel'

# RTMP 推流地址
RTMP_URL = 'rtmp://8.130.148.5:8080/live/hyxtest'
