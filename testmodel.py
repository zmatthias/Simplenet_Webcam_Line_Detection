from __future__ import print_function

import cv2
from simplenet import simplenet
import time
cap = cv2.VideoCapture(0)

WIDTH  = 8
HEIGHT = 6
LR = 1e-3

MODEL_NAME = 'simplenet.model'
model = simplenet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)

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
    #time.sleep(1)
    ret, webcamImage = cap.read()

    webcamImage = cv2.resize(webcamImage, (8, 6))
    webcamImage = cv2.cvtColor(webcamImage, cv2.COLOR_BGR2GRAY)
    webcamImage = cv2.flip(webcamImage, 1)
    prediction = model.predict([webcamImage.reshape(WIDTH,HEIGHT,1)])[0]
    printPrediction(prediction)

    cv2.imshow('frame', webcamImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

