import serial
import time

def return_pos():
    ser = serial.Serial("/dev/ttyACM0",115200)#portNo:端口地址；baudRate：波特率
    i = 0
    for i in range(8):
        print("four leg left")
        start_time=time.time()
        while True:
            ser.write('a'.encode())
            elapsed_time=time.time()-start_time
            if elapsed_time>=3:#distance是每次返回的值，逐帧更新
                break
    j=0
    for j in range(5):
        print("four leg forward")
        start_time=time.time()
        while True:
            ser.write('m'.encode())
            elapsed_time=time.time()-start_time
            if elapsed_time>=3:#distance是每次返回的值，逐帧更新
                break
    print("put down")
    start_time=time.time()
    while True:
        ser.write('p'.encode())
        elapsed_time=time.time()-start_time
        if elapsed_time>=3:#distance是每次返回的值，逐帧更新
            break
    exit()

if __name__ == '__main__':
    return_pos()