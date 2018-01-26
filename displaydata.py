import numpy as np
import cv2

def displayData(dataToDisplay):
    for data in dataToDisplay:
        image = data[0]
        choice = data[1]

        print("press Enter")
        raw_input()
        print(choice)
        image = cv2.resize(image, (512, 288),interpolation = cv2.INTER_NEAREST)
        cv2.imshow('test', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

set = np.load("evalSet.npy")
#set = np.load("trainSet.npy")
displayData(set)
