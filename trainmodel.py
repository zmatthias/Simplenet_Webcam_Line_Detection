import numpy as np
from simplenet import simplenet

WIDTH  = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 20

MODEL_NAME = 'simplenet.model'

model = simplenet(WIDTH,HEIGHT,LR)

savedTrainingData = np.load("trainingData.npy")

# 1500 images have been taken and are divided into two.
# the first 750 frames to train on
# and the last 750 frames to validate the training and recalibrate the weights

train = savedTrainingData[:-750]
test = savedTrainingData[-750:]

# each image is reshaped from a 60x80 matrix into a single array
X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)


Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
