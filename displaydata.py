import numpy as np
import cv2
import time

trainingData = np.load("validationData.npy")

def displayData(dataToDisplay):
    for data in dataToDisplay:
        image = data[0]
        choice = data[1]
        time.sleep(1)
        print(choice)
        image = cv2.resize(image, (400, 300),interpolation = cv2.INTER_NEAREST)
        cv2.imshow('test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

displayData(trainingData)
