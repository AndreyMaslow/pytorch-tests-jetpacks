#!/usr/bin/env bash

echo "[INFO] install python 3"
sudo apt-get install -y python3-dev python3-py python3-pytest python3-pip

echo "[INFO] installing PIL"
pip3 install Pillow

echo "[INFO] installing deps for pytorch"
sudo -H pip3 install Cython
sudo -H pip3 install scikit-image
sudo -H pip3 install scikit-build
sudo -H pip3 install numpy

echo "[INFO] installing pytorch"

echo "[INFO] cloning repo from http://github.com/pytorch/pytorch branch v 1.0.0"
git clone --single-branch --branch v1.0.0 http://github.com/pytorch/pytorch.git

cd pytorch

git submodule update --init
sudo pip3 install  setuptools
sudo pip3 install -r requirements.txt
python3 setup.py build_deps
sudo python3 setup.py install

echo "[INFO] installing torchvision"
sudo -H pip3 install torchvision

echo "[INFO] removing pytorch files"
cd ../
sudo rm -rf pytorch

