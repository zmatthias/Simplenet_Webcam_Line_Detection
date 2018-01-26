import numpy as np
import cv2
from random import shuffle
import os
import time

xDifference = 0
tiltsum = 0

trainSet = []
evalSet = []

def initCam():
    capInstance = cv2.VideoCapture(0)
    return capInstance

def ProcessImage(passedImage):
    processedImage = cv2.Canny(passedImage, threshold1=20, threshold2=100)
    processedImage = cv2.GaussianBlur(processedImage,(7,7),0)

    lines = cv2.HoughLinesP(processedImage, 1, np.pi/180, 10 ,np.array([]), 0, 20)
    return lines,processedImage

def DrawLines(lines,passedImage):
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:

                #Problem: Anfangs und Endpunkte der Linien werden geflippt, sodass der Endpunkt immer rechts ist
                #dadurch x1-x2 als Richungsvektor nicht moeglich
                if y2 >= y1:        #deflipping
                    x1_old = x1
                    x1 = x2
                    x2 = x1_old
                    y1_old = y1
                    y1 = y2
                    y2 = y1_old

                cv2.line(passedImage, (x1, y1), (x2, y2), (0, 255, 0), 10)
                global xDifference
                xDifference = x1-x2
    except:
        pass
    return passedImage

def getTilt(lines):
    try:
        for line in lines:
            global tiltsum
            tiltsum += xDifference
            print("xDifference:")
            print(xDifference)

        avgTilt = tiltsum / len(lines)
        tiltsum = 0
        print("avgTilt:")
        print(avgTilt)
        return avgTilt
    except:
        pass


def GetCamImage(capInstance):
    ret, camImage = capInstance.read()
    camImage = cv2.resize(camImage, (128, 72))
    camImage = cv2.cvtColor(camImage, cv2.COLOR_BGR2GRAY)
    camImage = cv2.flip(camImage, 1)
    return camImage

def ShowImage(image):
    image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()

def RecordTrainImages(n):
    capInstance = initCam()
    for i in range(n):
        image = GetCamImage(capInstance)
        lines = ProcessImage(image)
        print(lines)
        image = DrawLines(lines,image)
        ShowImage(image)
        steering = getTilt(lines)
        solution = [1,steering]
        trainSet.append([image, solution])
        print(solution)

def RecordEvalImages(n,solution):
    capInstance = initCam()
    for i in range(n):
        image = GetCamImage(capInstance)
        ShowImage(image)
        evalSet.append([image, solution])

def SaveTrainSet():
    shuffle(trainSet)
    np.save("trainSetFloat.npy", trainSet)

def SaveEvalSet():
    shuffle(evalSet)
    np.save("evalSetFloat.npy", evalSet)

def RecordTrainSet():
    RecordTrainImages(1000)
    try:
        os.remove("trainSetFloat.npy")
    except OSError:
        pass
    SaveTrainSet()

def RecordEvalSet():

    print("Press Enter to record 'left-eval'")
    time.sleep(3)
    solution = [1, 0, 0]
    RecordEvalImages(200, solution)

    print("Press Enter to record 'straight-eval'")
    time.sleep(3)
    solution = [0, 1, 0]
    RecordEvalImages(200, solution)

    print("Press Enter to record 'right-eval'")
    time.sleep(3)
    solution = [0, 0, 1]
    RecordEvalImages(200, solution)

    try:
        os.remove("evalSetFloat.npy")
    except OSError:
        pass

    SaveEvalSet()

RecordTrainSet()
#RecordEvalSet()

cv2.destroyAllWindows()