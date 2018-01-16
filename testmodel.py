from __future__ import print_function
import cv2
from simplenet import simplenet
capInstance = cv2.VideoCapture(0)

modelName = 'simplenet.model'
model = simplenet()
model.load(modelName)
inputWidth = 128
inputHeight = 72

def printPrediction(prediction):

    forwardCertainty = prediction[1]*100
    if (prediction[0] - prediction[2]) > 0:

        turningString = "left"
    else:
        turningString = "right"

    turningCertainty = abs((prediction[0] - prediction[2]))*100

    print("Forward: {}% \t \t \t \t \t Turn {}: {}% ".format(format(forwardCertainty, '.0f'), turningString, format(turningCertainty, '.0f')))

while(True):

    # Capture frame-by-frame
    ret, camImage = capInstance.read()

    camImage = cv2.resize(camImage, (inputWidth, inputHeight))
    camImage = cv2.cvtColor(camImage, cv2.COLOR_BGR2GRAY)
    camImage = cv2.flip(camImage, 1)
    prediction = model.predict([camImage.reshape(inputWidth,inputHeight,1)])[0]
    printPrediction(prediction)
    camImage = cv2.resize(camImage, (512, 288), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('frame', camImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
