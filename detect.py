import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import serial
import sys
sys.path.append('/usr/lib/python3/dist-packages')
sys.path.append('/usr/lib/python3/dist-packages/Jetson/GPIO')
import RPi.GPIO as GPIO
import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from back_pos import return_pos

TRIG = 13
ECHO = 18


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    print(view_img)
    view_img = False
    source = '0'
    weights = 'best.pt'
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det)<=0:
                print("not found")
                i = random.randint(1, 10)
                distance = 3
                try:
                    while True:
                        GPIO.output(TRIG, 0)
                        time.sleep(0.000002)

                        GPIO.output(TRIG, 1)
                        time.sleep(0.00001)
                        GPIO.output(TRIG, 0)

                        while GPIO.input(ECHO) == 0:
                            a = 0
                        time1 = time.time()
                        while GPIO.input(ECHO) == 1:
                            a = 1
                        time2 = time.time()

                        during = time2 - time1
                        faraway = during * 343 / 2 * 100
                        print("Faraway", during * 343 / 2 * 100)
                        if during != 0:
                            break

                except KeyboardInterrupt:
                    GPIO.cleanup()

                print("forward")
                start_time = time.time()
                while True:
                    ser.write('f'.encode())
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 3:
                        break

                if i % 2 == 0:
                    print("slightly left")
                    start_time = time.time()
                    while True:
                        ser.write('z'.encode())
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 10 * distance:  # distance是每次返回的值，逐帧更新
                            break
                else:
                    print("slightly right")
                    start_time = time.time()
                    while True:
                        ser.write('y'.encode())
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= 10 * distance:  # distance是每次返回的值，逐帧更新
                            break

                print("slightly down")
                start_time = time.time()
                while True:
                    ser.write('q'.encode())
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 5:
                        break

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = []
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # xywh[0]  x
                    # xywh[1]  y
                    # xywh[2]  w
                    # xywh[3]  h

                    try:
                        while True:
                            GPIO.output(TRIG, 0)
                            time.sleep(0.000002)

                            GPIO.output(TRIG, 1)
                            time.sleep(0.00001)
                            GPIO.output(TRIG, 0)

                            while GPIO.input(ECHO) == 0:
                                a = 0
                            time1 = time.time()
                            while GPIO.input(ECHO) == 1:
                                a = 1
                            time2 = time.time()

                            during = time2 - time1
                            faraway = during * 343 / 2 * 100
                            print("Faraway", during * 343 / 2 * 100)
                            if during != 0:
                                break

                    except KeyboardInterrupt:
                        GPIO.cleanup()

                    print("x ", xywh[0])
                    print("new round")
                    # 1 mid
                    if xywh[0] < 0.72 and xywh[0] > 0.57:
                        print("x ", xywh[0])
                        start_time = time.time()
                        print("forward")
                        while True:
                            ser.write('f'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 5:
                                break

                        if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                            start_time = time.time()
                            print("catch")
                            while True:
                                ser.write('h'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            return_pos()
                            exit()
                        start_time = time.time()
                        print("slightly down")
                        while True:
                            ser.write('q'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 3:
                                break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()


                    # 1 left
                    elif xywh[0] < 0.57:
                        print("x ", xywh[0])
                        start_time = time.time()
                        print("slightly turn left")
                        while True:
                            ser.write('z'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 3:
                                break

                        if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                            start_time = time.time()
                            print("catch")
                            while True:
                                ser.write('h'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            return_pos()
                            exit()

                        start_time = time.time()
                        print("forward")
                        while True:
                            ser.write('f'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 3:
                                break

                        start_time = time.time()
                        print("slightly down")
                        while True:
                            ser.write('q'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 6:
                                break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                                # 2 mid
                        if xywh[0] < 0.71 and xywh[0] > 0.57:
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 3:
                                    break

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 3:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                        # 2 left
                        elif xywh[0] < 0.57:
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("slightly turn left")
                            while True:
                                ser.write('z'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                        # 2 right
                        else:
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("slightly turn right")
                            while True:
                                ser.write('y'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                    # 1 right
                    else:  # xywh[0]>0.72
                        print("x ", xywh[0])
                        start_time = time.time()
                        print("slightly turn right")
                        while True:
                            ser.write('y'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 5:
                                break
                        if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                            start_time = time.time()
                            print("catch")
                            while True:
                                ser.write('h'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            return_pos()
                            exit()

                        start_time = time.time()
                        print("forward")
                        while True:
                            ser.write('f'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 5:
                                break

                        start_time = time.time()
                        print("slightly down")
                        while True:
                            ser.write('q'.encode())
                            elapsed_time = time.time() - start_time
                            if elapsed_time >= 6:
                                break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                                # 2 mid
                        if xywh[0] < 0.71 and xywh[0] > 0.57:
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                        # 2 left
                        elif xywh[0] < 0.57:
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("slightly turn left")
                            while True:
                                ser.write('z'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                        # 2 right
                        else:
                            print("x ", xywh[0])
                            start_time = time.time()
                            print("slightly turn right")
                            while True:
                                ser.write('y'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break
                            if xywh[2] > 0.49 or xywh[3] > 0.5:  # ideal condition
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()

                            start_time = time.time()
                            print("forward")
                            while True:
                                ser.write('f'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 5:
                                    break

                            start_time = time.time()
                            print("slightly down")
                            while True:
                                ser.write('q'.encode())
                                elapsed_time = time.time() - start_time
                                if elapsed_time >= 6:
                                    break
                            # detect catch
                            print("faraway ", faraway)
                            if faraway < 16 and faraway > 14:  # if xywh[3]<0.47 and xywh[3]>0.42
                                start_time = time.time()
                                print("catch")
                                while True:
                                    ser.write('h'.encode())
                                    elapsed_time = time.time() - start_time
                                    if elapsed_time >= 6:
                                        break
                                return_pos()
                                exit()
                    print(xywh)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')


    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)
    ser = serial.Serial("/dev/ttyACM0",115200)#portNo; baudRate

    with torch.no_grad():
        detect()
