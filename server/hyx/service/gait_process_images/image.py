import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 加载预训练的DeepLab模型
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
people_class = 15

# 评估模型
model.eval()
print("Model has been loaded.")

blur = torch.FloatTensor([[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]) / 16.0

# 使用GPU（如果支持）以获得更好的性能
if torch.cuda.is_available():
    model.to('cuda')
    blur = blur.to('cuda')

# 应用预处理（归一化）
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 创建分割掩码的函数
def makeSegMask(img):
    # 将图像转换为PIL图像
    img = Image.fromarray(img)

    # 构建输入tensor
    input_tensor = preprocess(img).unsqueeze(0)

    # 使用GPU（如果支持）以获得更好的性能
    if torch.cuda.is_available():
        input_tensor = input_tensor.to('cuda')

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    segmentation = output.argmax(0)

    bgOut = output[0:1][:][:]
    a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

    people = segmentation.eq(torch.ones_like(segmentation).long().fill_(people_class)).float()

    people.unsqueeze_(0).unsqueeze_(0)

    for i in range(3):
        people = F.conv2d(people, blur, stride=1, padding=1)

    # 激活函数结合掩码 - F.hardtanh(a * b)
    combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5))))
    combined_mask = combined_mask.expand(1, 3, -1, -1)

    res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

    return res


def process_images(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            continue

        print(f"Processing {image_name}")
        img = cv2.imread(image_path)
        if img is None:
            continue

        mask = makeSegMask(img)

        # 应用阈值以将掩码转换为二值图
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 保存处理后的图像
        save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(save_path, thresh)


if __name__ == '__main__':
    image_dir = "/home/hyx/Desktop/server/hyx/service/in"
    output_dir = "/home/hyx/Desktop/server/hyx/service/out/image"
    process_images(image_dir, output_dir)
