import cv2
import numpy as np
from random import shuffle
import time

trainingData = []

cap = cv2.VideoCapture(0)


def GetWebcamImage():
    ret, webcamImage = cap.read()
    webcamImage = cv2.resize(webcamImage, (80, 60))
    webcamImage = cv2.cvtColor(webcamImage, cv2.COLOR_BGR2GRAY)
    webcamImage = cv2.flip(webcamImage, 1)
    return webcamImage

def ShowImage(image):
    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()

def RecordImages(n):
    for i in range(n):
        image = GetWebcamImage()
        ShowImage(image)
        trainingData.append([image, direction])
        print(direction)

def SaveTrainingData():
    shuffle(trainingData)
    np.save("trainingData.npy", trainingData)

print("3 seconds until recording 'left'")
time.sleep(3)
direction = [1,0,0]
RecordImages(500)

print("3 seconds until recording 'straight'")
time.sleep(3)
direction = [0,1,0]
RecordImages(500)

print("3 seconds until recording 'right'")
time.sleep(3)
direction = [0,0,1]
RecordImages(500)

SaveTrainingData()
cv2.destroyAllWindows()