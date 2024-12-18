# Capturer based on YOLOv7

This is a project to Capturer using YOLOv7.

## Video Presentation
[![Hexapod Robot](https://res.cloudinary.com/marcomontalbano/image/upload/v1716715642/video_to_markdown/images/youtube--DM378_XFm8g-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/DM378_XFm8g "Hexapod Robot")

## Web Code Repository

- Integrated into [Capturer 🤗](https://github.com/juyujing/Capturer) using [yolov7](https://github.com/WongKinYiu/yolov7). Try to use the latest version of [Capturer](https://github.com/juyujing/Capturer)！

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name capturer -it -v your_datasets_path/:/datasets/ -v your_code_path/:/capturer --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install environment required packages
pip install seaborn thop

# pip install cpaturer required packages
pip install -r requirements.txt

# go to code folder
cd /capturer
```
</details>

## Model Parameter Download
- [.pt parameter file download](https://drive.google.com/file/d/1cYjeel8Tn4-sg-7VWGWGWIrWprRs_7OQ/view?usp=sharing)
- [.onnx parameter file download link](https://drive.google.com/file/d/10R5ECCBh2b9J1TIRrAcoTmRHNhMhBM-E/view?usp=sharing)

## Run

Deploy the ActionSet File Config.ini on 24-Way steering geer control board and boot it.

Deploy the File connect_arduino_jetsonnano.py on Adruino Uno and boot it.

Deploy the Code Folder on Jetson Nano and run the following command:

``` shell
python detect.py
```

## Defalut Settings

- weights: best.pt && best.onnx
- source: 0 (default camera)
- cfg: cfg/training/yovov7-snail.yaml
- data: data/snail.yaml
- img_size: 640

## 📚 License
This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
For commercial purposes, please contact [yj2012@hw.ac.uk] to obtain explicit permission.
