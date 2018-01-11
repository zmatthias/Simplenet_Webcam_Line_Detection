import cv2
import numpy as np
from random import shuffle
import time

trainingData = []
validationData = []

cap = cv2.VideoCapture(1)

def GetWebcamImage():
    ret, webcamImage = cap.read()
    webcamImage = cv2.resize(webcamImage, (8, 6))
    webcamImage = cv2.cvtColor(webcamImage, cv2.COLOR_BGR2GRAY)
    webcamImage = cv2.flip(webcamImage, 1)
    return webcamImage

def ShowImage(image):
    image = cv2.resize(image, (400, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()


def RecordTrainingImages(n):
    for i in range(n):
        image = GetWebcamImage()
        ShowImage(image)
        trainingData.append([image, direction])
        print(direction)

def RecordValidationImages(n):
    for i in range(n):
        image = GetWebcamImage()
        ShowImage(image)
        validationData.append([image, direction])
        print(direction)

def SaveTrainingData():
    shuffle(trainingData)
    np.save("trainingData.npy", trainingData)

def SaveValidationData():
    shuffle(validationData)
    np.save("validationData.npy", validationData)

print("3 seconds until recording 'left-train'")
time.sleep(3)
direction = [1,0,0]
RecordTrainingImages(1000)

print("3 seconds until recording 'straight-train'")
time.sleep(3)
direction = [0,1,0]
RecordTrainingImages(1000)

print("3 seconds until recording 'right-train'")
time.sleep(3)
direction = [0,0,1]
RecordTrainingImages(1000)

SaveTrainingData()

print("3 seconds until recording 'left-val'")
time.sleep(3)
direction = [1,0,0]
RecordValidationImages(1000)

print("3 seconds until recording 'straight-val'")
time.sleep(3)
direction = [0,1,0]
RecordValidationImages(1000)

print("3 seconds until recording 'right-val'")
time.sleep(3)
direction = [0,0,1]
RecordValidationImages(1000)

SaveValidationData()

cv2.destroyAllWindows()