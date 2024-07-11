# import cv2
# import numpy as np
# import torch
# import os
# import config
# import torch
# from torchvision import transforms
# from PIL import Image
# import os
# from scipy.spatial.distance import cosine
# import sys
# import torch
# import os
# from torchvision import transforms
# from PIL import Image
#
# # 将主目录添加到 sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
# from model_pack.model2 import detect_and_draw
# from service.gait.model_pack import Model
#
# from torchvision import transforms
# from PIL import Image
#
#
# class GaitFeatureExtractor:
#     def __init__(self, model_path):
#         self.model_pack = Model(hidden_dim=256, restore_iter=100, model_name='gaitset', checkpoint_dir=model_path, use_cuda = False)
#         self.model_pack.load(restore_iter=100)
#         self.device = self.model_pack.device
#
#     # def load_model(self, model_path):
#     #     model_pack = torch.load(model_path)
#     #     model_pack.eval()
#     #     return model_pack
#
#     def preprocess_image(self, image_path):
#         preprocess = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485], std=[0.229]),
#         ])
#         image = Image.open(image_path).convert('L')
#         image_tensor = preprocess(image)
#         return image_tensor
#
#     def extract_features(self, image_dir):
#         image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if
#                        img.endswith(('.png', '.jpg', '.jpeg'))]
#         image_tensors = [self.preprocess_image(img_path) for img_path in image_paths]
#
#         # 如果图片少于100张，重复最后一张图片直到满足100张
#         while len(image_tensors) < 100:
#             image_tensors.append(image_tensors[-1])
#
#         # 只保留前100张图片
#         image_tensors = image_tensors[:100]
#
#         # 将图片堆叠成一个tensor
#         # image_tensors = torch.stack(image_tensors).to(self.device)
#         # 将图片堆叠成一个tensor，形状为 [batch_size, sequence_length, channels, height, width]
#         image_tensors = torch.stack(image_tensors).to(self.device).unsqueeze(0)
#         print(image_tensors.shape)
#
#         with torch.no_grad():
#             feature, _ = self.model_pack.encoder(image_tensors.unsqueeze(0))
#             feature = feature.cpu().numpy().squeeze()
#
#         return feature
#
# def extract_gait_features_for_person(image_dir, model_path):
#     extractor = GaitFeatureExtractor(model_path)
#     features = extractor.extract_features(image_dir)
#     return features.mean(axis=0)  # 返回平均特征向量
#
# # 主函数
# def main():
#     image_dir = '/home/hyx/Desktop/server/hyx/data/srr'  # 修改为您的图像文件夹路径
#     model_path = '/home/hyx/Desktop/server/hyx/service/weights/gait'  # 修改为您的模型检查点文件夹路径
#     checkpoint_iter = 100  # 修改为您希望加载的检查点迭代次数
#
#     image_dir1 = "/home/hyx/Desktop/server/hyx/data/srr"
#     image_dir2 = "/home/hyx/Desktop/server/hyx/data/image"
#     model_path = "/home/hyx/Desktop/server/hyx/service/weights/gait"
#
#     feature1 = extract_gait_features_for_person(image_dir1, model_path)
#     feature2 = extract_gait_features_for_person(image_dir2, model_path)
#
#     # 比较特征
#     similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
#     print(f"Similarity between person 1 and person 2: {similarity}")
#
# if __name__ == "__main__":
#     main()
import os
import cv2
import numpy as np
import torch
import sys
from PIL import Image
from scipy.spatial.distance import cosine
from torchvision import transforms

# 将主目录添加到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
from model.model2 import detect_and_draw
from service.modelg.model import Model


class GaitFeatureExtractor:
    def __init__(self, model_path):
        self.model = Model(hidden_dim=256, restore_iter=100, model_name='gaitset', checkpoint_dir=model_path, use_cuda=False)
        self.model.load(restore_iter=100)
        self.device = self.model.device

    def preprocess_image(self, image_path):
        preprocess = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        image = Image.open(image_path).convert('L')
        image_tensor = preprocess(image)
        return image_tensor

    def extract_features(self, image_dir):
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        image_tensors = [self.preprocess_image(img_path) for img_path in image_paths]

        while len(image_tensors) < 100:
            image_tensors.append(image_tensors[-1])

        image_tensors = image_tensors[:100]
        image_tensors = torch.stack(image_tensors).to(self.device).unsqueeze(0)
        # print(image_tensors.shape)

        with torch.no_grad():
            feature, _ = self.model.encoder(image_tensors.unsqueeze(0))
            feature = feature.cpu().numpy().squeeze()

        return feature


def extract_gait_features_for_person(image_dir, model_path):
    extractor = GaitFeatureExtractor(model_path)
    features = extractor.extract_features(image_dir)
    return features.mean(axis=0)


def main(base_image_dir):
    # # base_image_dir = '/home/hyx/Desktop/server/hyx/data/srr'
    # model_path = '/home/hyx/Desktop/server/hyx/service/weights/gait'
    # base_feature = extract_gait_features_for_person(base_image_dir, model_path)
    #
    # parent_dir = '/home/hyx/Desktop/server/hyx/data'
    # subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d != 'srr']
    #
    # max_similarity = -1
    # most_similar_dir = None
    #
    # for subdir in subdirs:
    #     current_feature = extract_gait_features_for_person(subdir, model_path)
    #     similarity = np.dot(base_feature, current_feature) / (np.linalg.norm(base_feature) * np.linalg.norm(current_feature))
    #     print(f"Similarity with {os.path.basename(subdir)}: {similarity}")
    #
    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         most_similar_dir = subdir
    #
    # print(f"Most similar directory: {most_similar_dir} with similarity {max_similarity}")
    # base_image_dir = '/home/hyx/Desktop/server/hyx/data/srr'
    compare_image_dir = '/home/hyx/Desktop/server/hyx/data'
    model_path = '/home/hyx/Desktop/server/hyx/service/weights/gait'

    base_features = extract_gait_features_for_person(base_image_dir, model_path)

    max_similarity = -1
    most_similar_dir = None

    for dir_name in os.listdir(compare_image_dir):
        dir_path = os.path.join(compare_image_dir, dir_name)
        if os.path.isdir(dir_path) and dir_path != base_image_dir:
            compare_features = extract_gait_features_for_person(dir_path, model_path)
            similarity = np.dot(base_features, compare_features) / (
                        np.linalg.norm(base_features) * np.linalg.norm(compare_features))
            #print(f"Similarity with {dir_name}: {similarity}")

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_dir = dir_name

    print(f"Most similar directory: {most_similar_dir} with similarity {max_similarity}")
    return most_similar_dir,max_similarity


# if __name__ == "__main__":
#     main(base_image_dir)
