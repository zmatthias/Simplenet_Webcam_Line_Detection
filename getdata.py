import cv2
import numpy as np
from random import shuffle
import time
import os

trainSet = []
evalSet = []

def initCam():
    capInstance = cv2.VideoCapture(0)
    return capInstance

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

def RecordTrainImages(n,solution):
    capInstance = initCam()
    for i in range(n):
        image = GetCamImage(capInstance)
        ShowImage(image)
        trainSet.append([image, solution])

def RecordEvalImages(n,solution):
    capInstance = initCam()
    for i in range(n):
        image = GetCamImage(capInstance)
        ShowImage(image)
        evalSet.append([image, solution])

def SaveTrainSet():
    shuffle(trainSet)
    np.save("trainSet.npy", trainSet)

def SaveEvalSet():
    shuffle(evalSet)
    np.save("evalSet.npy", evalSet)

def RecordTrainSet():

    print("3 seconds until recording 'left-train'")
    time.sleep(3)
    solution = [1, 0, 0]
    RecordTrainImages(1000,solution)
    print("3 seconds until recording 'straight-train'")
    time.sleep(3)
    solution = [0, 1, 0]
    RecordTrainImages(1000,solution)
    print("3 seconds until recording 'right-train'")
    time.sleep(3)
    solution = [0, 0, 1]
    RecordTrainImages(1000,solution)
    try:
        os.remove("trainSet.npy")
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
        os.remove("evalSet.npy")
    except OSError:
        pass

    SaveEvalSet()

RecordTrainSet()
#RecordEvalSet()

cv2.destroyAllWindows()