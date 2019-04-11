# Pytorch-Tests-Jetpacks

Repository of pytorch mtcnn model speed measuring. 
There is a problem that mtcnn on jetpack 4.2 and 3.3 predictions take different time on jetson tx2.

## Dependencies
opencv 3.4.1, pytorch (v1.0.0), numpy (use ./installs.sh)

## Model 
This repository use pretrained mtcnn face detector from https://github.com/TropComplique/mtcnn-pytorch

## Usage
git clone https://github.com/AndreyMaslow/pytorch-tests-jetpacks

cd pytorch-tests-jetpacks

./installs.sh

python3 speed_test.py
