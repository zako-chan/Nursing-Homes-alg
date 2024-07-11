import os
import oss2
import config

def upload_to_oss(file_path):
    auth = oss2.Auth(config.OSS_ACCESS_KEY_ID, config.OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, config.OSS_ENDPOINT, config.OSS_BUCKET_NAME)

    # 获取文件名
    file_name = os.path.basename(file_path)

    # 上传文件
    bucket.put_object_from_file(file_name, file_path)

    # 获取文件URL
    url = f"http://{config.OSS_BUCKET_NAME}.{config.OSS_ENDPOINT}/{file_name}"
    return url
