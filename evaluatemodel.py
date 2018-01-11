from simplenet import simplenet
import numpy as np
import cv2


MODEL_NAME = 'simplenet.model'

WIDTH  = 8
HEIGHT = 6
LR = 1e-3

model = simplenet(WIDTH,HEIGHT,LR)

savedValidationData = np.load("trainingData.npy")

def predictData(savedValidationData):
    for data in savedValidationData:
        prediction = model.predict([data[0].reshape(WIDTH, HEIGHT, 1)])[0]
        print(prediction)
        image = cv2.resize(data[0], (400, 300),interpolation = cv2.INTER_NEAREST)


        cv2.imshow('test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


test_x = np.array([i[0] for i in savedValidationData]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = np.array([i[1] for i in savedValidationData])

score = model.evaluate(test_x,test_y)
print(score)

#while(True):
#    predictData(savedValidationData)