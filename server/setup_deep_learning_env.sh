#!/bin/bash

# 创建 TensorFlow 虚拟环境
echo "Creating TensorFlow virtual environment..."
conda create -n tensorflow python=3.6 -y

# 激活虚拟环境
echo "Activating TensorFlow virtual environment..."
conda activate tensorflow

# 安装 OpenCV
echo "Installing OpenCV..."
pip install opencv-python==3.4.4.19
pip install opencv-contrib-python==3.4.4.19

# 验证 OpenCV 安装
echo "Verifying OpenCV installation..."
python -c "import cv2; print(cv2.__version__)"

# 安装 TensorFlow
echo "Installing TensorFlow..."
pip install tensorflow==1.12.0

# 验证 TensorFlow 安装
echo "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print(tf.__version__)"

# 安装 Keras
echo "Installing Keras..."
pip install keras==2.2.4

# 验证 Keras 安装
echo "Verifying Keras installation..."
python -c "import keras"

# 安装 dlib 所需的系统依赖
echo "Installing system dependencies for dlib..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libgtk-3-dev libboost-all-dev

# 安装 dlib
echo "Installing dlib..."
pip install dlib

# 验证 dlib 安装
echo "Verifying dlib installation..."
python -c "import dlib"

# 安装 face_recognition
echo "Installing face_recognition..."
pip install face_recognition

# 验证 face_recognition 安装
echo "Verifying face_recognition installation..."
python -c "import face_recognition"

echo "Deep learning development environment setup is complete."

