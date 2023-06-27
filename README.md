## salsanext-adapter

### environment 
* A100 GPU 8
* cuda 11.1
* cudnn 8.3
* torch 1.9
* python 3.8.13
* settting docker 
```
docker image pull zombbie/cuda11.1-cudnn8-ubuntu20.04:v1.0 
docker run --name vit-adapter-kitti-train --gpus all --shm-size=1024gb -it -v /jyjeon/vit-adapter-kitti/:/vit-adapter-kitti -e TZ=Asia
/Seoul zombbie/cuda11.1-cudnn8-ubuntu20.04:v1.0
```
* conda create
```
$ conda init
$ conda activate
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
$ pip install timm==0.4.12
$ pip install mmdet==2.22.0 # for Mask2Former
$ pip install mmsegmentation==0.20.2
$ ln -s ../detection/ops ./ # link 
$ cd ops & sh make.sh # compile deformable attention
```
*  lib error 발생 시
```
$ apt-get update
$ apt-get -y install libgl1-mesa-glx
```


* distutils error 발생 시 
```
$ pip install setuptools==59.5.0 # AttributedError: module 'distutils' has no attribute 'version'
```

